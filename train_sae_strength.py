"""
SAE强度预测器训练脚本 - 重构版本

这个脚本用于训练一个神经网络来预测SAE（稀疏自编码器）特征的激活强度。
主要功能：
1. 使用VLLM采样器生成文本
2. 训练强度预测器来控制SAE特征激活
3. 使用PPO算法进行强化学习训练
4. 支持分布式训练和详细的指标记录

重构后的结构：
- TrainingConfig: 配置管理
- DataProcessor: 数据处理
- ModelManager: 模型初始化和管理
- BatchProcessor: 批次数据处理
- MetricsLogger: 指标记录和日志
- StrengthTrainer: 主训练类
"""

from models.vllm_sampler import VllmSampler
from models.strengths_predictor import StrengthsPredictor
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from utils.data_utils import load_jsonl, load_txt
from datasets import Dataset
from torch.utils.data import DataLoader
from utils.reward_utils import reward_func
from utils.utils import process_sequences, tokenize_fn, compute_approx_kl, compute_reward, compute_entropy
from algorithms.reinforce_pp_baseline import compute_reinforce_plus_plus_baseline_outcome_advantage
from losses.vanilla import compute_policy_loss
import os
import debugpy
import swanlab
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

DEBUG = False
# debugpy.listen(5678)
# debugpy.wait_for_client()


@dataclass
class TrainingConfig:
    """训练配置类"""
    # 数据相关
    train_data_path: str
    test_data_path: str
    train_prompt_path: str
    test_prompt_path: str
    batch_size: int = 1
    
    # 模型相关
    model: str = "meta-llama/Llama-2-7b-hf"
    tensor_parallel_size: int = 1
    gpu_util: str = "0.8"
    max_model_length: int = 2048
    max_input_len: int = 1024
    max_output_len: int = 1024
    
    # SAE相关
    sae_path: Optional[str] = None
    sae_release: Optional[str] = None
    sae_id: Optional[str] = None
    feature_idxs: str = None
    max_activations: str = None
    
    # 采样相关
    temperature: float = 0.6
    top_p: float = 0.95
    n_samples_per_prompt: int = 1
    logprobs: int = 128000
    
    # 训练相关
    lr: float = 1e-4
    num_epochs: int = 10
    clip_range: float = 0.2
    clip_range_low: float = 0.2
    clip_range_high: float = 0.2
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    kl_coef: float = 0.01
    reward_clip_range: Tuple[float, float] = (-10, 10)
    
    # 保存和评估
    save_dir: str = "./checkpoints"
    save_interval: int = 100
    eval_interval: int = 100


class DataProcessor:
    """数据处理类"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def load_and_prepare_data(self) -> Tuple[DataLoader, DataLoader]:
        """加载和准备训练、测试数据"""
        # 加载prompt模板
        train_prompt = load_txt(self.config.train_prompt_path)
        test_prompt = load_txt(self.config.test_prompt_path)
        
        # 加载数据
        train_data = list(load_jsonl(self.config.train_data_path))
        for data in train_data:
            data['problem'] = data['problem'] + '\n' + train_prompt
            
        test_data = list(load_jsonl(self.config.test_data_path))
        for data in test_data:
            data['problem'] = data['problem'] + '\n' + test_prompt
        
        # 创建Dataset和DataLoader
        train_dataset = Dataset.from_list(train_data)
        test_dataset = Dataset.from_list(test_data)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        
        return train_loader, test_loader


class ModelManager:
    """模型管理类"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.vllm_sampler = None
        self.strength_predictor = None
        self.optimizer = None
        self.predictor_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        
    def initialize_models(self) -> Tuple[VllmSampler, StrengthsPredictor, torch.optim.Optimizer]:
        """初始化所有模型"""
        # 配置VLLM参数
        llm_kwargs = {
            'model': self.config.model,
            'gpu_memory_utilization': float(self.config.gpu_util),
            'tensor_parallel_size': self.config.tensor_parallel_size,
            'max_model_len': self.config.max_model_length,
            'trust_remote_code': True,
            'enforce_eager': True if self.config.sae_path or self.config.sae_release or self.config.sae_id else False,
        }
        
        sae_kwargs = {
            'path': self.config.sae_path,
            'release': self.config.sae_release,
            'id': self.config.sae_id,
            'feature_idxs': self.config.feature_idxs,
            'max_activations': self.config.max_activations,
        }
        
        sampling_kwargs = {
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'max_tokens': self.config.max_output_len,
            'n': 1,
            'logprobs': self.config.logprobs - 1,
        }
        
        # 初始化VLLM采样器
        self.vllm_sampler = VllmSampler(llm_kwargs, sae_kwargs, sampling_kwargs)
        
        # 初始化强度预测器
        hidden_state_dim = self.vllm_sampler.get_hidden_state_size()
        feature_num = len(self.config.feature_idxs.split(','))
        self.strength_predictor = StrengthsPredictor(
            hidden_state_dim, 
            feature_num, 
            hidden_dim=hidden_state_dim//4
        )
        
        # 移动到指定设备
        self.strength_predictor.to(self.predictor_device)
        print(f"StrengthsPredictor placed on: {self.predictor_device}")
        
        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.strength_predictor.parameters(), lr=self.config.lr)
        
        return self.vllm_sampler, self.strength_predictor, self.optimizer


class MetricsLogger:
    """指标记录类"""
    
    def __init__(self, config: TrainingConfig, debug: bool = False):
        self.config = config
        self.debug = debug
        
        if not self.debug:
            swanlab.init(
                project="SAE-Strength-Training",
                experiment_name=f"strength_predictor_{config.model.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}",
                config=self._get_swanlab_config(),
                description="Training strength predictor for SAE feature control"
            )
    
    def _get_swanlab_config(self) -> Dict[str, Any]:
        """获取SwanLab配置"""
        return {
            "model": self.config.model,
            "learning_rate": self.config.lr,
            "batch_size": self.config.batch_size,
            "num_epochs": self.config.num_epochs,
            "max_input_len": self.config.max_input_len,
            "max_output_len": self.config.max_output_len,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "clip_range": self.config.clip_range,
            "feature_idxs": self.config.feature_idxs,
            "max_activations": self.config.max_activations,
        }
    
    def log_strength_stats(self, strengths_values: np.ndarray, step: int):
        """记录强度预测统计信息"""
        if not self.debug:
            swanlab.log({
                "strength_stats/mean": np.mean(strengths_values),
                "strength_stats/std": np.std(strengths_values),
                "strength_stats/min": np.min(strengths_values),
                "strength_stats/max": np.max(strengths_values),
            }, step=step)
    
    def log_training_metrics(self, metrics: Dict[str, float], step: int):
        """记录训练指标"""
        if not self.debug:
            swanlab.log({f"train/{k}": v for k, v in metrics.items()}, step=step)
    
    def log_validation_metrics(self, metrics: Dict[str, float], step: int):
        """记录验证指标"""
        if not self.debug:
            swanlab.log({f"validation/{k}": v for k, v in metrics.items()}, step=step)
    
    def log_epoch_metrics(self, metrics: Dict[str, float], step: int):
        """记录epoch级别指标"""
        if not self.debug:
            swanlab.log({f"epoch/{k}": v for k, v in metrics.items()}, step=step)
    
    def finish(self):
        """完成日志记录"""
        if not self.debug:
            swanlab.finish()


class BatchProcessor:
    """批次数据处理类"""
    
    def __init__(self, config: TrainingConfig, vllm_sampler: VllmSampler):
        self.config = config
        self.vllm_sampler = vllm_sampler
        self.tokenizer = vllm_sampler.get_tokenizer()
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
    
    def process_batch_outputs(self, outputs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str], List[float], torch.Tensor]:
        """处理批次输出，返回处理后的数据"""
        sequences = []
        resp_sequences = []
        resp_lens = []
        responses = []
        sae_logprobs = []
        sae_all_logprobs = []
        
        # 计算最大长度
        max_input_len = max(len(output.prompt_token_ids) for output in outputs)
        max_output_len = max(len(output.outputs[0].token_ids) for output in outputs)
        
        # 处理每个输出
        for output in outputs:
            resp_lens.append(len(list(output.outputs[0].token_ids)))
            
            # 填充序列
            prompt_ids = [self.pad_token_id] * (max_input_len - len(output.prompt_token_ids)) + list(output.prompt_token_ids)
            resp_ids = list(output.outputs[0].token_ids) + [self.pad_token_id] * (max_output_len - len(output.outputs[0].token_ids))
            
            sequences.append(prompt_ids + resp_ids)
            resp_sequences.append(resp_ids)
            responses.append(output.outputs[0].text)
            
            # 处理logprobs
            token_logprobs = []
            all_token_logprobs = []
            
            for i, token_data in enumerate(output.outputs[0].logprobs):
                if token_data is not None:
                    sampled_token_id = output.outputs[0].token_ids[i]
                    token_logprobs.append(token_data[sampled_token_id].logprob)
                    
                    # 获取所有token的logprobs
                    token_logprob_tensor = torch.full((self.config.logprobs,), float('-inf'))
                    for idx, (token_id, logprob_obj) in enumerate(token_data.items()):
                        if idx < self.config.logprobs:
                            token_logprob_tensor[idx] = logprob_obj.logprob
                    all_token_logprobs.append(token_logprob_tensor)
                else:
                    raise ValueError(f"token_data is None for output {output}")
            
            sae_logprobs.append(token_logprobs)
            sae_all_logprobs.append(torch.stack(all_token_logprobs))
        
        # 转换为tensor
        sequences = torch.tensor(sequences).to("cpu")
        resp_sequences = torch.tensor(resp_sequences).to("cpu")
        
        # 处理序列
        sequences, attention_masks, action_masks = process_sequences(
            sequences, max_input_len, self.eos_token_id, self.pad_token_id
        )
        
        # 处理sae_logprobs
        max_len = action_masks.shape[1]
        processed_sae_logprobs = []
        for logprobs in sae_logprobs:
            if len(logprobs) > max_len:
                raise ValueError(f"logprobs长度大于max_len: {len(logprobs)} > {max_len}")
            else:
                processed_sae_logprobs.append(logprobs + [0.0] * (max_len - len(logprobs)))
        
        sae_logprobs_tensor = torch.tensor(processed_sae_logprobs)
        
        # 处理all_logprobs
        padded_logprobs = []
        for seq_logprobs in sae_all_logprobs:
            if len(seq_logprobs) < max_len:
                padding = torch.full((max_len - len(seq_logprobs), self.config.logprobs), float('-inf'))
                seq_logprobs = torch.cat([seq_logprobs, padding], dim=0)
            padded_logprobs.append(seq_logprobs)
        
        all_logprobs_tensor = torch.stack(padded_logprobs)
        
        return (sequences, attention_masks, action_masks, responses, resp_lens, 
                sae_logprobs_tensor, all_logprobs_tensor)


class StrengthTrainer:
    """强度预测器训练类"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.logger = MetricsLogger(config, DEBUG)
        self.data_processor = DataProcessor(config)
        self.model_manager = ModelManager(config)
        self.batch_processor = None
        
        # 初始化模型
        self.vllm_sampler, self.strength_predictor, self.optimizer = self.model_manager.initialize_models()
        self.batch_processor = BatchProcessor(config, self.vllm_sampler)
        
        # 获取数据
        self.train_loader, self.test_loader = self.data_processor.load_and_prepare_data()
        
        # 计算特征数量
        self.feature_num = len(config.feature_idxs.split(','))
        
    def compute_batch_metrics(self, sequences: torch.Tensor, attention_masks: torch.Tensor, 
                            action_masks: torch.Tensor, responses: List[str], 
                            prompts: List[str], answers: List[str], 
                            sae_logprobs: torch.Tensor, all_logprobs_tensor: torch.Tensor,
                            resp_lens: List[float]) -> Dict[str, Any]:
        """计算批次指标"""
        # 准备重复的prompts和answers
        repeated_prompts = []
        repeated_answers = []
        index = []
        for i in range(len(prompts)):
            for _ in range(self.config.n_samples_per_prompt):
                index.append(i)
                repeated_prompts.append(prompts[i])
                repeated_answers.append(answers[i])
        
        # 计算rewards
        rewards = reward_func(responses, repeated_prompts, repeated_answers)
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards)
        rewards = rewards.to('cpu')
        
        # 计算reference logprobs
        ref_logprobs = self.vllm_sampler.get_logprobs(sequences, attention_masks)
        ref_logprobs = ref_logprobs.to('cpu')
        ref_logprobs = ref_logprobs[:, :-1]
        ref_logprobs = ref_logprobs[:, -action_masks.shape[1]:] * action_masks.float()
        
        # 计算KL penalty
        sae_logprobs = sae_logprobs.to('cpu')
        kl_penalty = compute_approx_kl(sae_logprobs, ref_logprobs)
        batch_kl_penalty = torch.mean(kl_penalty).item()
        
        # 计算序列熵
        response_logprobs = all_logprobs_tensor[:, -action_masks.shape[1]:, :].to('cpu')
        sequence_entropies = compute_entropy(response_logprobs, action_masks, temperature=self.config.temperature)
        avg_sequence_entropy = torch.mean(sequence_entropies).item()
        
        # 计算token级别rewards
        token_level_rewards = compute_reward(
            rewards, self.config.kl_coef, kl_penalty,
            action_mask=action_masks, reward_clip_range=self.config.reward_clip_range,
        )
        
        # 计算advantages
        advantages, returns = compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards, action_masks, index
        )
        
        return {
            'rewards': rewards,
            'kl_penalty': kl_penalty,
            'batch_kl_penalty': batch_kl_penalty,
            'sequence_entropies': sequence_entropies,
            'avg_sequence_entropy': avg_sequence_entropy,
            'token_level_rewards': token_level_rewards,
            'advantages': advantages,
            'resp_lens': resp_lens,
            'batch_rewards': torch.mean(rewards).item(),
            'avg_response_length': np.mean(resp_lens),
            'response_length_std': np.std(resp_lens)
        }
    
    def train_step(self, batch) -> Dict[str, float]:
        """执行一个训练步骤"""
        self.strength_predictor.train()
        prompts = batch['problem']
        answers = batch['answer']
        
        # 获取hidden states
        hidden_states = self.vllm_sampler.get_last_token_hidden_state(prompts)
        hidden_states = hidden_states.to(self.model_manager.predictor_device)
        
        # 预测strengths
        predicted_strengths = self.strength_predictor(hidden_states)
        strengths_values = predicted_strengths.detach().cpu().numpy()
        
        # 生成输出
        outputs = self.vllm_sampler.generate(
            prompts, strengths_values.tolist(), self.config.n_samples_per_prompt
        )
        
        # 处理批次输出
        (sequences, attention_masks, action_masks, responses, resp_lens, 
         sae_logprobs, all_logprobs_tensor) = self.batch_processor.process_batch_outputs(outputs)
        
        # 计算指标
        metrics = self.compute_batch_metrics(
            sequences, attention_masks, action_masks, responses, 
            prompts, answers, sae_logprobs, all_logprobs_tensor, resp_lens
        )
        
        # 计算损失
        advantages = metrics['advantages'][:, :self.feature_num]
        advantages = advantages.to(self.model_manager.predictor_device)
        
        new_predicted_strengths = self.strength_predictor(hidden_states)
        new_predicted_strengths = new_predicted_strengths.repeat_interleave(
            self.config.n_samples_per_prompt, dim=0
        )
        old_predicted_strengths = predicted_strengths.detach().repeat_interleave(
            self.config.n_samples_per_prompt, dim=0
        )
        
        strength_mask = torch.ones_like(advantages, dtype=torch.bool)
        
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
            old_log_prob=old_predicted_strengths,
            log_prob=new_predicted_strengths,
            advantages=advantages,
            response_mask=strength_mask,
            cliprange=self.config.clip_range,
            cliprange_low=self.config.clip_range_low,
            cliprange_high=self.config.clip_range_high,
            clip_ratio_c=self.config.clip_ratio_c,
            loss_agg_mode=self.config.loss_agg_mode,
        )
        
        # 反向传播
        self.optimizer.zero_grad()
        pg_loss.backward()
        self.optimizer.step()
        
        return {
            'policy_loss': pg_loss.item(),
            'kl_penalty': metrics['batch_kl_penalty'],
            'rewards_mean': metrics['batch_rewards'],
            'advantages_mean': torch.mean(advantages).item(),
            'ppo_kl': ppo_kl,
            'sequence_entropy': metrics['avg_sequence_entropy'],
            'response_length_mean': metrics['avg_response_length'],
            'response_length_std': metrics['response_length_std'],
            'strengths_values': strengths_values
        }
    
    def validate(self) -> Dict[str, float]:
        """执行验证"""
        self.strength_predictor.eval()
        correct_num = 0
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Validation"):
                prompts = batch['problem']
                answers = batch['answer']
                
                hidden_states = self.vllm_sampler.get_last_token_hidden_state(prompts)
                hidden_states = hidden_states.to(self.model_manager.predictor_device)
                predicted_strengths = self.strength_predictor(hidden_states)
                
                outputs = self.vllm_sampler.generate(
                    prompts, predicted_strengths.detach().cpu().numpy().tolist(), 1
                )
                
                responses = [output.outputs[0].text for output in outputs]
                rewards = reward_func(responses, prompts, answers)
                
                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards)
                rewards = rewards.to("cpu")
                correct_num += (rewards > 0).sum().item()
        
        val_duration = time.time() - val_start_time
        accuracy = correct_num / len(self.test_loader.dataset) * 100
        
        return {
            'accuracy': accuracy,
            'duration': val_duration
        }
    
    def save_model(self, step: int):
        """保存模型"""
        save_path = os.path.join(self.config.save_dir, f"strength_predictor_{step}.pth")
        torch.save(self.strength_predictor.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    def train(self):
        """主训练循环"""
        global_step = 0
        
        for epoch in range(self.config.num_epochs):
            # 初始化epoch统计
            total_metrics = {
                'loss': 0.0, 'pg_loss': 0.0, 'kl_penalty': 0.0, 
                'rewards': 0.0, 'advantages': 0.0, 'sequence_entropy': 0.0, 
                'response_length': 0.0
            }
            num_batches = 0
            epoch_start_time = time.time()
            
            for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{self.config.num_epochs}"):
                # 训练步骤
                step_metrics = self.train_step(batch)
                
                # 记录强度统计
                self.logger.log_strength_stats(step_metrics['strengths_values'], global_step)
                
                # 记录训练指标
                train_metrics = {k: v for k, v in step_metrics.items() if k != 'strengths_values'}
                self.logger.log_training_metrics(train_metrics, global_step)
                
                # 累计统计
                total_metrics['loss'] += step_metrics['policy_loss']
                total_metrics['pg_loss'] += step_metrics['policy_loss']
                total_metrics['kl_penalty'] += step_metrics['kl_penalty']
                total_metrics['rewards'] += step_metrics['rewards_mean']
                total_metrics['advantages'] += step_metrics['advantages_mean']
                total_metrics['sequence_entropy'] += step_metrics['sequence_entropy']
                total_metrics['response_length'] += step_metrics['response_length_mean']
                num_batches += 1
                global_step += 1
                
                # 验证
                if self.test_loader and global_step % self.config.eval_interval == 0:
                    val_metrics = self.validate()
                    val_metrics['epoch'] = epoch + 1
                    self.logger.log_validation_metrics(val_metrics, global_step)
                    print(f"Epoch {epoch+1}, Validation accuracy: {val_metrics['accuracy']:.1f}%, Duration: {val_metrics['duration']:.2f}s")
                
                # 保存模型
                if global_step % self.config.save_interval == 0:
                    self.save_model(global_step)
            
            # 记录epoch统计
            epoch_duration = time.time() - epoch_start_time
            avg_metrics = {k: v / num_batches for k, v in total_metrics.items()}
            avg_metrics.update({
                'duration': epoch_duration,
                'samples_per_second': len(self.train_loader.dataset) / epoch_duration,
                'number': epoch + 1
            })
            
            self.logger.log_epoch_metrics(avg_metrics, global_step)
            print(f"Epoch {epoch+1}/{self.config.num_epochs}, Average Loss: {avg_metrics['loss']:.4f}, Duration: {epoch_duration:.2f}s")
        
        # 完成训练
        self.logger.log_training_metrics({'completed': 1, 'total_steps': global_step}, global_step)
        print("Training completed!")
        print(f"Total training steps: {global_step}")
        
        self.logger.finish()


def create_config_from_args(args) -> TrainingConfig:
    """从命令行参数创建配置对象"""
    return TrainingConfig(
        # 数据相关
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        train_prompt_path=args.train_prompt_path,
        test_prompt_path=args.test_prompt_path,
        batch_size=args.batch_size,
        
        # 模型相关
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_util=args.gpu_util,
        max_model_length=args.max_model_length,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        
        # SAE相关
        sae_path=args.sae_path,
        sae_release=args.sae_release,
        sae_id=args.sae_id,
        feature_idxs=args.feature_idxs,
        max_activations=args.max_activations,
        
        # 采样相关
        temperature=args.temperature,
        top_p=args.top_p,
        n_samples_per_prompt=args.n_samples_per_prompt,
        logprobs=args.logprobs,
        
        # 训练相关
        lr=args.lr,
        num_epochs=args.num_epochs,
        clip_range=args.clip_range,
        clip_range_low=args.clip_range_low,
        clip_range_high=args.clip_range_high,
        clip_ratio_c=args.clip_ratio_c,
        loss_agg_mode=args.loss_agg_mode,
        kl_coef=args.kl_coef,
        reward_clip_range=tuple(args.reward_clip_range),
        
        # 保存和评估
        save_dir=args.save_dir,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
    )


def main(args):
    """主函数 - 使用重构后的类结构"""
    # 创建配置对象
    config = create_config_from_args(args)
    
    # 创建训练器并开始训练
    trainer = StrengthTrainer(config)
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default=None)
    parser.add_argument("--test_data_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--tensor_parallel_size', type=int, default=1)
    parser.add_argument("--gpu_util", type=str, default="0.8")
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument('--max_model_length', type=int, default=2048)
    parser.add_argument('--max_input_len', type=int, default=1024)
    parser.add_argument("--max_output_len", type=int, default=1024)
    parser.add_argument("--n_samples_per_prompt", type=int, default=1)
    parser.add_argument("--sae_path", type=str, default=None)
    parser.add_argument("--sae_release", type=str, default=None)
    parser.add_argument("--sae_id", type=str, default=None)
    parser.add_argument("--feature_idxs", type=str, default=None)
    parser.add_argument("--max_activations", type=str, default=None)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--clip_range_low", type=float, default=0.2)
    parser.add_argument("--clip_range_high", type=float, default=0.2)
    parser.add_argument("--clip_ratio_c", type=float, default=3.0)
    parser.add_argument("--loss_agg_mode", type=str, default="token-mean")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for strength predictor")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Path to save the trained model")
    parser.add_argument("--save_interval", type=int, default=100, help="Save interval")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL penalty coefficient")
    parser.add_argument("--vocab_size", type=int, default=128000, help="Vocabulary size")
    parser.add_argument("--logprobs", type=int, default=128000, help="Logprobs")
    parser.add_argument("--train_prompt_path", type=str, default=None, help="Path to prompt")
    parser.add_argument("--test_prompt_path", type=str, default=None, help="Path to prompt")
    parser.add_argument("--eval_interval", type=int, default=100, help="Evaluation interval")
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    main(args)