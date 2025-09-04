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
        
    def generate(self, prompts: list[str], strengths_list: list[list[float]], samples_per_prompt: int):
        outputs = []
        for prompt, strengths in zip(prompts, strengths_list):
            sae_hooks = []
            if self.sae:
                sae_hooks.append((self.lm_model.model.layers[self.sae.cfg.hook_layer], self.get_multi_intervention_hook_with_strengths(strengths=strengths)))
            with add_hooks([], sae_hooks):
                output = self.llm.generate(
                    [prompt]*samples_per_prompt,  # vLLM的generate方法期望接收一个列表，即使只有一个prompt
                    self.sampling_params,
                )
            outputs.extend(output)  # 使用extend而不是append，因为output本身就是一个列表
        return outputs

    def get_last_token_hidden_state(self, prompts: list[str])->torch.Tensor:
        hidden_states = []
        
        tokenizer = self.tokenizer
        for i, prompt in enumerate(prompts):
            # 对每个prompt进行tokenization
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.model.device)
            attention_mask = inputs["attention_mask"].to(self.model.device)
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            
            # 通过self.model前向传播获取hidden states
            with torch.no_grad():
                model_outputs = self.model(input_ids, position_ids=position_ids, attention_mask=attention_mask, output_hidden_states=True)
                # 获取最后一个token的hidden state
                # hidden_states是一个tuple，最后一层的hidden state是最后一个元素
                last_layer_hidden_states = model_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_dim]
                last_hidden_state = last_layer_hidden_states[0, -1, :]  # [hidden_dim]
                hidden_states.append(last_hidden_state)
        # (N, d_h)
        hidden_states = torch.stack(hidden_states)
        return hidden_states
    
    def get_logprobs(self, sequences: torch.Tensor, attention_masks: torch.Tensor, temperature: float=1.0)->torch.Tensor:
        """
        使用 self.model 获取每个 token 的 log probabilities
        
        Args:
            sequences: [batch_size, seq_len] 的 token ids
            
        Returns:
            logprobs: [batch_size, seq_len] 的 log probabilities
        """
        sequences = sequences.to(self.model.device)
        rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
        rolled_sequences = rolled_sequences.to(self.model.device)
        attention_masks = attention_masks.to(self.model.device)
        # 创建 attention mask (假设没有padding或者padding token是0)
        # 创建 position ids
        position_ids = attention_masks.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_masks == 0, 1)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=sequences,
                attention_mask=attention_masks,
                position_ids=position_ids
            )
            
            # 获取 logits: [batch_size, seq_len, vocab_size]
            logits = outputs.logits
            
            # 计算 log probabilities
            log_probs = log_probs_from_logits(logits, rolled_sequences, temperature)  # [batch_size, seq_len, vocab_size]
            
            # 获取每个位置实际 token 的 log probability
            # 使用 gather 来选择对应 token 的 log prob
            # token_log_probs = torch.gather(
            #     log_probs, 
            #     dim=-1, 
            #     index=sequences.unsqueeze(-1)  # [batch_size, seq_len, 1]
            # ).squeeze(-1)  # [batch_size, seq_len]
            
        return log_probs

    def get_hidden_state_size(self)->int:
        return self.model.config.hidden_size
    
    def get_num_layers(self)->int:
        return self.model.config.num_hidden_layers
    
    def get_tokenizer(self):
        return self.tokenizer