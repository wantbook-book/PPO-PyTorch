from models.vllm_sampler import VllmSampler
from models.strengths_predictor import StrengthsPredictor
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
from utils.data_utils import load_jsonl
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
import psutil
DEBUG=True
debugpy.listen(5678)
debugpy.wait_for_client()

def main(args):
    # 初始化 SwanLab
    if not DEBUG:
        swanlab.init(
            project="SAE-Strength-Training",
            experiment_name=f"strength_predictor_{args.model.split('/')[-1]}_{time.strftime('%Y%m%d_%H%M%S')}",
            config={
                "model": args.model,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "max_input_len": args.max_input_len,
                "max_output_len": args.max_output_len,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "clip_range": args.clip_range,
                "feature_idxs": args.feature_idxs,
                "max_activations": args.max_activations,
            },
            description="Training strength predictor for SAE feature control"
        )
    
    llm_kwargs = {
        'model': args.model,
        'gpu_memory_utilization': float(args.gpu_util),
        'tensor_parallel_size': args.tensor_parallel_size,
        'max_model_len': args.max_model_length,
        'trust_remote_code': True,
        'enforce_eager': True if args.sae_path or args.sae_release or args.sae_id else False,  # Required for SAE hooks to work properly
    }
    sae_kwargs = {
        'path': args.sae_path,
        'release': args.sae_release,
        'id': args.sae_id,
        'feature_idxs': args.feature_idxs,
        'max_activations': args.max_activations,
    }
    sampling_kwargs = {
        'temperature': args.temperature,
        'top_p': args.top_p,
        'max_tokens': args.max_output_len,
        'n': 1,
        'logprobs': True,  # 获取所有logprobs用于计算损失
    }
    vllm_sampler = VllmSampler(llm_kwargs, sae_kwargs, sampling_kwargs)
    hidden_state_dim = vllm_sampler.get_hidden_state_size()
    feature_num = len(sae_kwargs['feature_idxs'].split(','))
    strength_predictor = StrengthsPredictor(hidden_state_dim, feature_num, hidden_dim=hidden_state_dim//4)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(strength_predictor.parameters(), lr=args.lr)
    # 将strength_predictor放在GPU0
    predictor_device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    strength_predictor.to(predictor_device)
    print(f"StrengthsPredictor placed on: {predictor_device}")

    train_data = list(load_jsonl(args.train_data_path))
    test_data = list(load_jsonl(args.test_data_path))
    train_data = Dataset.from_list(train_data)
    test_data = Dataset.from_list(test_data)
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    tokenizer = vllm_sampler.get_tokenizer()
    pad_token_id, eos_token_id = tokenizer.pad_token_id, tokenizer.eos_token_id
    
    # 训练循环
    best_loss = float('inf')
    global_step = 0
    
    for epoch in range(args.num_epochs):
        strength_predictor.train()
        total_loss = 0.0
        total_pg_loss = 0.0
        total_kl_penalty = 0.0
        total_rewards = 0.0
        total_advantages = 0.0
        total_sequence_entropy = 0.0
        total_response_length = 0.0
        num_batches = 0
        epoch_start_time = time.time()
        
        for batch in tqdm(train_data_loader, desc=f"Training Epoch {epoch+1}/{args.num_epochs}"):
            prompts = batch['problem']
            answers = batch['answer']
            
            # 获取prompt的hidden states (从GPU1的vLLM)
            hidden_states = vllm_sampler.get_last_token_hidden_state(prompts)
            # 将hidden states从GPU1转移到GPU0给strength_predictor使用
            hidden_states = hidden_states.to(predictor_device)
            
            # 预测strengths
            predicted_strengths = strength_predictor(hidden_states)
            strengths_values = predicted_strengths.detach().cpu().numpy()
            
            if not DEBUG:
                # 记录强度预测统计信息
                swanlab.log({
                    "strength_stats/mean": np.mean(strengths_values),
                    "strength_stats/std": np.std(strengths_values),
                    "strength_stats/min": np.min(strengths_values),
                    "strength_stats/max": np.max(strengths_values),
                }, step=global_step)
            
            # 使用预测的strengths生成输出
            outputs = vllm_sampler.generate(prompts, strengths_values.tolist(), args.n_samples_per_prompt)
            
            sequences = []
            resp_sequences = []
            resp_lens = []
            responses = []
            sae_logprobs = []
            sae_logits = []
            max_input_len, max_output_len = 0, 0
            for output in outputs:
                max_input_len = max(max_input_len, len(output.prompt_token_ids))
                max_output_len = max(max_output_len, len(output.outputs[0].token_ids))

            for output in outputs:
                resp_lens.append(len(list(output.outputs[0].token_ids)))
                prompt_ids = [pad_token_id] * (max_input_len - len(output.prompt_token_ids)) + list(output.prompt_token_ids)
                resp_ids = list(output.outputs[0].token_ids) + [pad_token_id] * (max_output_len - len(output.outputs[0].token_ids))
                sequences.append(prompt_ids + resp_ids)
                resp_sequences.append(resp_ids)
                responses.append(output.outputs[0].text)
                # 提取选中token的logprobs
                # 可以用output.outputs[0].logprobs: list[dict[int, Logprob]]
                token_logprobs = []
                for token_data in output.outputs[0].logprobs:
                    if token_data is not None:
                        sampled_token_id = output.outputs[0].token_ids[len(token_logprobs)]
                        token_logprobs.append(token_data[sampled_token_id].logprob)
                    else:
                        token_logprobs.append(0.0)
                sae_logprobs.append(token_logprobs)
                sae_logits.append(output.outputs[0].logits)
            
            # 计算响应的 token 长度
            avg_response_length = np.mean(resp_lens)
            response_length_std = np.std(resp_lens)
            
            sequences = torch.tensor(sequences)
            sequences = sequences.to("cpu")
            resp_sequences = torch.tensor(resp_sequences)
            resp_sequences = resp_sequences.to("cpu")
            sequences, attention_masks, action_masks = process_sequences(sequences, max_input_len, eos_token_id, pad_token_id)
            
            index = []
            repeated_prompts = []
            repeated_answers = []
            for i in range(len(prompts)):
                for _ in range(args.n_samples_per_prompt):
                    index.append(i)
                    repeated_prompts.append(prompts[i])
                    repeated_answers.append(answers[i])
            # 计算rewards
            rewards = reward_func(responses, repeated_prompts, repeated_answers)
            # 确保rewards是tensor并在CPU上
            if not isinstance(rewards, torch.Tensor):
                rewards = torch.tensor(rewards)
            rewards = rewards.to('cpu')
            
            # 计算logprobs
            ref_logprobs = vllm_sampler.get_logprobs(sequences, attention_masks)
            ref_logprobs = ref_logprobs[:, :-1]
            action_masks = action_masks.to(ref_logprobs.device)
            ref_logprobs = ref_logprobs[:, -action_masks.shape[1]:] * action_masks.float()
            
            # 处理sae_logprobs，确保长度一致
            max_len = action_masks.shape[1]
            processed_sae_logprobs = []
            for logprobs in sae_logprobs:
                if len(logprobs) > max_len:
                    # 不可能比他长啊
                    raise ValueError(f"logprobs长度大于max_len: {len(logprobs)} > {max_len}")
                    # processed_sae_logprobs.append(logprobs[:max_len])
                else:
                    processed_sae_logprobs.append(logprobs + [0.0] * (max_len - len(logprobs)))
            sae_logprobs = torch.tensor(processed_sae_logprobs)
            
            # 计算序列级别的熵 (sequence level entropy)
            # 使用 sae_logprobs 来计算熵，因为这是我们实际生成时的概率分布
            # 将 log probabilities 转换为 probabilities
            # 计算每个序列的熵: H = -sum(p * log(p))
            # 只计算有效 token 的熵（使用 action_masks）
            sae_logits = torch.tensor(sae_logits)
            sae_logits = sae_logits[:, -action_masks.shape[1]:] * action_masks.float()
            sae_logits = sae_logits.to('cpu')
            sequence_entropies = compute_entropy(sae_logits, action_masks, temperature=args.temperature)
            # 归一化：除以有效 token 数量
            avg_sequence_entropy = torch.mean(sequence_entropies).item()
            
            # 确保所有tensor在同一设备上进行计算
            sae_logprobs = sae_logprobs.to('cpu')
            ref_logprobs = ref_logprobs.to('cpu')
            action_masks = action_masks.to('cpu')
            
            # 计算token level kl penalty
            kl_penalty = compute_approx_kl(sae_logprobs, ref_logprobs)
            # 只考虑有效token的，不需要嘛？openrlfh是全部平均的
            batch_kl_penalty = torch.mean(kl_penalty).item()
            
            # token_level_rewards计算，reward加在最后一个token
            token_level_rewards = compute_reward(
                rewards,
                args.kl_coef,
                kl_penalty,
                action_mask=action_masks,
                reward_clip_range=args.reward_clip_range,
            )
            token_level_rewards = token_level_rewards.to('cpu')

            index = torch.tensor(index).to('cpu')
            advantages, returns = compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards, action_masks, index)
            
            # 确保所有用于loss计算的tensor都在CPU上
            advantages = advantages.to(predictor_device)
            sae_logprobs = sae_logprobs.to(predictor_device)
            ref_logprobs = ref_logprobs.to(predictor_device)
            action_masks = action_masks.to(predictor_device)

            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = compute_policy_loss(
                old_log_prob=ref_logprobs,
                log_prob=sae_logprobs,
                advantages=advantages,
                response_mask=action_masks,
                cliprange=args.clip_range,
                cliprange_low=args.clip_range_low,
                cliprange_high=args.clip_range_high,
                clip_ratio_c=args.clip_ratio_c,
                loss_agg_mode=args.loss_agg_mode,
            )
            
            # 计算批次统计信息
            batch_rewards = torch.mean(rewards).item()
            batch_advantages = torch.mean(advantages).item()
            
            if not DEBUG:
                # 记录详细的训练指标
                swanlab.log({
                    "train/policy_loss": pg_loss.item(),
                    "train/kl_penalty": batch_kl_penalty,
                    "train/rewards_mean": batch_rewards,
                    "train/advantages_mean": batch_advantages,
                    "train/ppo_kl": ppo_kl,
                    "train/learning_rate": args.lr,
                    "train/sequence_entropy": avg_sequence_entropy,
                    "train/response_length_mean": avg_response_length,
                    "train/response_length_std": response_length_std,
                }, step=global_step)
            
            # 反向传播更新strength_predictor
            optimizer.zero_grad()
            pg_loss.backward()
            optimizer.step()
            
            # 累计统计信息
            total_loss += pg_loss.item()
            total_pg_loss += pg_loss.item()
            total_kl_penalty += batch_kl_penalty
            total_rewards += batch_rewards
            total_advantages += batch_advantages
            total_sequence_entropy += avg_sequence_entropy
            total_response_length += avg_response_length
            num_batches += 1
            global_step += 1
            
        # 计算 epoch 统计信息
        epoch_duration = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        avg_pg_loss = total_pg_loss / num_batches
        avg_kl_penalty = total_kl_penalty / num_batches
        avg_rewards = total_rewards / num_batches
        avg_advantages = total_advantages / num_batches
        avg_sequence_entropy = total_sequence_entropy / num_batches
        avg_response_length = total_response_length / num_batches
        
        if not DEBUG:
            # 记录 epoch 级别指标
            swanlab.log({
                "epoch/avg_loss": avg_loss,
                "epoch/avg_policy_loss": avg_pg_loss,
                "epoch/avg_kl_penalty": avg_kl_penalty,
                "epoch/avg_rewards": avg_rewards,
                "epoch/avg_advantages": avg_advantages,
                "epoch/avg_sequence_entropy": avg_sequence_entropy,
                "epoch/avg_response_length": avg_response_length,
                "epoch/duration": epoch_duration,
                "epoch/samples_per_second": len(train_data) / epoch_duration,
                "epoch/number": epoch + 1,
            }, step=global_step)
        
        print(f"Epoch {epoch+1}/{args.num_epochs}, Average Loss: {avg_loss:.4f}, Duration: {epoch_duration:.2f}s")
        
        # 评估和保存模型
        # if avg_loss < best_loss:
        #     best_loss = avg_loss
        #     swanlab.log({"train/best_loss": best_loss}, step=global_step)
        #     if args.save_path:
        #         os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
        #         torch.save(strength_predictor.state_dict(), args.save_path)
        #         print(f"Model saved to {args.save_path}")
        #         # 记录模型保存事件
        #         swanlab.log({"model/saved": 1, "model/save_epoch": epoch + 1}, step=global_step)
        
        # 验证集评估
        if test_data_loader:
            strength_predictor.eval()
            correct_num = 0
            val_start_time = time.time()
            
            with torch.no_grad():
                for batch in tqdm(test_data_loader, desc="Validation"):
                    # 简化的验证过程，只计算预测准确性
                    prompts = batch['problem']
                    answers = batch['answer']
                    hidden_states = vllm_sampler.get_last_token_hidden_state(prompts)
                    # 将hidden states从GPU1转移到GPU0
                    hidden_states = hidden_states.to(predictor_device)
                    predicted_strengths = strength_predictor(hidden_states)
                    # 这里可以添加更具体的验证指标
                    outputs = vllm_sampler.generate(prompts, predicted_strengths.detach().cpu().numpy().tolist())
                    responses = []
                    for output in outputs:
                        responses.append(output.outputs[0].text)
                    rewards = reward_func(responses, prompts, answers)
                    # 确保rewards是tensor并在CPU上
                    if not isinstance(rewards, torch.Tensor):
                        rewards = torch.tensor(rewards)
                    rewards = rewards.to("cpu")
                    correct_num += (rewards > 0).sum().item()
            
            val_duration = time.time() - val_start_time
            accuracy = correct_num / len(test_data) * 100
            
            if not DEBUG:
                # 记录验证指标
                swanlab.log({
                    "validation/accuracy": accuracy,
                    "validation/duration": val_duration,
                    "validation/epoch": epoch + 1,
                }, step=global_step)
            
            print(f"Epoch {epoch+1}, Validation accuracy: {accuracy:.1f}%, Duration: {val_duration:.2f}s")
    
    if not DEBUG:
        # 记录训练完成
        swanlab.log({
            "training/completed": 1,
            "training/total_steps": global_step,
        }, step=global_step)
    
    print("Training completed!")
    print(f"Total training steps: {global_step}")
    
    if not DEBUG:
        # 完成实验
        swanlab.finish()


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
    parser.add_argument("--save_path", type=str, default="./checkpoints/strength_predictor.pth", help="Path to save the trained model")
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")
    parser.add_argument("--kl_coef", type=float, default=0.01, help="KL penalty coefficient")
    args = parser.parse_args()
    main(args)