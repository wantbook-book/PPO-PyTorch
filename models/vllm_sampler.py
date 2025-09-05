from vllm import LLM, SamplingParams
from sae_lens import SAE
from utils.sae_utils import add_hooks, get_multi_intervention_hook
import torch
from functools import partial
from transformers import AutoModelForCausalLM
from utils.utils import log_probs_from_logits
class VllmSampler:
    def __init__(self, llm_kwargs, sae_kwargs, sampling_kwargs) -> None:
        self.llm = LLM(**llm_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            llm_kwargs["model"],
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda:2"
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.sampling_params = SamplingParams(**sampling_kwargs)
        if sae_kwargs.get("path", None):
            self.sae = SAE.load_from_pretrained(path=sae_kwargs.pop("path"))
        elif sae_kwargs.get("release", None) and sae_kwargs.get("id", None):
            self.sae, _, _ = SAE.from_pretrained(
                release=sae_kwargs.pop("release"),
                sae_id=sae_kwargs.pop("id")
            )
        else:
            self.sae = None
        if self.sae:
            self.lm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            feature_idxs = sae_kwargs.get("feature_idxs")
            feature_idxs = list(map(int, feature_idxs.split(',')))
            max_activations = sae_kwargs.get("max_activations")
            max_activations = list(map(float, max_activations.split(',')))
            # strengths 需要运行时确定，先用用一个partial得到函数
            self.get_multi_intervention_hook_with_strengths = partial(get_multi_intervention_hook, sae=self.sae, feature_idxs=feature_idxs, max_activations=max_activations)
        else:
            self.get_multi_intervention_hook_with_strengths = None
        
    def generate(self, prompts: list[str], strengths_list: list[list[float]], samples_per_prompt: int=1):
        outputs = []
        for prompt, strengths in zip(prompts, strengths_list):
            sae_hooks = []
            if self.sae:
                sae_hooks.append((self.lm_model.model.layers[self.sae.cfg.hook_layer], self.get_multi_intervention_hook_with_strengths(strengths=strengths)))
            with add_hooks([], sae_hooks):
                output = self.llm.generate(
                    [prompt]*samples_per_prompt,  # vLLM的generate方法期望接收一个列表，即使只有一个prompt
                    self.sampling_params,
                    use_tqdm=False
                )
            outputs.extend(output)  # 使用extend而不是append，因为output本身就是一个列表
        return outputs

    def get_last_token_hidden_state(self, prompts: list[str], batch_size: int=8)->torch.Tensor:
        """
        获取每个prompt最后一个token的hidden state
        采用分批处理来避免OOM问题
        
        Args:
            prompts: 输入的prompt列表
            batch_size: 每批处理的prompt数量，默认为8
            
        Returns:
            hidden_states: [num_prompts, hidden_dim] 的tensor
        """
        hidden_states = []
        tokenizer = self.tokenizer
        total_prompts = len(prompts)
        
        # 分批处理prompts
        for i in range(0, total_prompts, batch_size):
            end_idx = min(i + batch_size, total_prompts)
            batch_prompts = prompts[i:end_idx]
            
            # 批量tokenization
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)
            
            # 创建position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            
            # 通过self.model前向传播获取hidden states
            with torch.no_grad():
                model_outputs = self.model(
                    input_ids, 
                    position_ids=position_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True
                )
                
                # 获取最后一层的hidden states
                last_layer_hidden_states = model_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
                
                # 对每个样本获取最后一个有效token的hidden state
                batch_hidden_states = []
                for j in range(last_layer_hidden_states.shape[0]):
                    # 找到最后一个非padding token的位置
                    seq_len = attention_mask[j].sum().item()
                    last_hidden_state = last_layer_hidden_states[j, seq_len-1, :]  # [hidden_dim]
                    batch_hidden_states.append(last_hidden_state.cpu())  # 移动到CPU
                
                hidden_states.extend(batch_hidden_states)
                
                # 显式删除大tensor以释放内存
                del model_outputs, last_layer_hidden_states
                torch.cuda.empty_cache()
            
            # 删除batch数据
            del input_ids, attention_mask, position_ids
        
        # 合并所有批次的结果
        hidden_states = torch.stack(hidden_states)
        return hidden_states
    
    def get_logprobs(self, sequences: torch.Tensor, attention_masks: torch.Tensor, temperature: float=1.0, batch_size: int=4)->torch.Tensor:
        """
        使用 self.model 获取每个 token 的 log probabilities
        采用分批处理来避免OOM问题
        
        Args:
            sequences: [batch_size, seq_len] 的 token ids
            attention_masks: [batch_size, seq_len] 的 attention masks
            temperature: 温度参数
            batch_size: 每批处理的样本数量，默认为4
            
        Returns:
            logprobs: [batch_size, seq_len] 的 log probabilities
        """
        total_samples = sequences.shape[0]
        all_log_probs = []
        
        # 分批处理
        for i in range(0, total_samples, batch_size):
            end_idx = min(i + batch_size, total_samples)
            batch_sequences = sequences[i:end_idx]
            batch_attention_masks = attention_masks[i:end_idx]
            
            # 移动到GPU
            batch_sequences = batch_sequences.to(self.model.device)
            batch_attention_masks = batch_attention_masks.to(self.model.device)
            
            # 准备rolled sequences（用于计算log_probs）
            rolled_sequences = torch.roll(batch_sequences, shifts=-1, dims=1)
            
            # 创建 position ids
            position_ids = batch_attention_masks.long().cumsum(-1) - 1
            position_ids.masked_fill_(batch_attention_masks == 0, 1)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch_sequences,
                    attention_mask=batch_attention_masks,
                    position_ids=position_ids
                )
                
                # 获取 logits: [batch_size, seq_len, vocab_size]
                logits = outputs.logits
                
                # 立即移动到CPU以释放GPU内存
                logits = logits.to('cpu')
                rolled_sequences = rolled_sequences.to('cpu')
                
                # 计算 log probabilities
                batch_log_probs = log_probs_from_logits(logits, rolled_sequences, temperature)
                all_log_probs.append(batch_log_probs)
                
                # 显式删除大tensor以释放内存
                del logits, outputs
                torch.cuda.empty_cache()  # 清理GPU缓存
            
            # 显式删除batch数据
            del batch_sequences, batch_attention_masks, position_ids, rolled_sequences
        
        # 合并所有批次的结果
        log_probs = torch.cat(all_log_probs, dim=0)
        return log_probs

    def get_hidden_state_size(self)->int:
        return self.model.config.hidden_size
    
    def get_num_layers(self)->int:
        return self.model.config.num_hidden_layers
    
    def get_tokenizer(self):
        return self.tokenizer