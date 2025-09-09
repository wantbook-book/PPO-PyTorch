import sys
import argparse
sys.path.append('..')
from vllm import LLM, SamplingParams
from utils.sae_utils import add_hooks, get_multi_intervention_hook, GlobalSAE
from sae_lens import SAE
import numpy as np
import time
from transformer_lens import HookedTransformer
import torch
def main(args):
    llm_kwagrs = {
        'model': args.model,
        'tensor_parallel_size': args.tensor_parallel_size,
        'gpu_util': args.gpu_util,
        'max_model_len': args.max_model_length,
        'trust_remote_code': True,
        'enforce_eager': True,
        'dtype': torch.bfloat16
    }
    llm = LLM(**llm_kwagrs)
    sampling_params = SamplingParams(
        top_p=args.top_p,
        temperature=args.temperature,
        max_tokens=args.max_output_len,
    )
    sae_kwargs = {
        'path': args.sae_path,
        'release': args.sae_release,
        'id': args.sae_id,
        'feature_idxs': args.feature_idxs,
        'max_activations': args.max_activations,
        'strengths': args.strengths,
    }
    sae = None
    if sae_kwargs.get("path", None):
        sae = SAE.load_from_pretrained(path=sae_kwargs.pop("path"))
    elif sae_kwargs.get("release", None) and sae_kwargs.get("id", None):
        sae, _, _ = SAE.from_pretrained(
            release=sae_kwargs.pop("release"),
            sae_id=sae_kwargs.pop("id")
        )
    sae_hooks = []
    feature_idxs = []
    max_activations = []
    strengths = []
    if sae:
        feature_idxs = list(map(int, sae_kwargs.pop("feature_idxs").split(',')))
        max_activations = list(map(float, sae_kwargs.pop("max_activations").split(',')))
        strengths = list(map(float, sae_kwargs.pop("strengths").split(',')))
        lm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        sae_hooks.append((lm_model.model.layers[sae.cfg.hook_layer], get_multi_intervention_hook(sae, feature_idxs, max_activations, strengths)))
    
    test_prompts = [
        '请简要介绍一下人工智能的发展历史。'
    ]

    if sae:
        print('======vllm with sae hooks start=====')    
        start = time.time()
        output_tokens = []
        with add_hooks([], sae_hooks):
            outputs = llm.generate(test_prompts, sampling_params)
        for output in outputs:
            output_tokens.append(len(output.outputs[0].token_ids))
        vllm_duration = time.time() - start
        output_tokens = np.array(output_tokens)
        print(f"vllm with sae hooks average output tokens: {output_tokens.mean()}")
        print(f"vllm with sae hooks cost time: {vllm_duration}s")
        print('======vllm with sae hooks end=====')    
    else:
        print("No SAE model loaded, skipping vllm with sae hooks test.")

    if sae:
        model = HookedTransformer.from_pretrained_no_processing(
            args.model,
            dtype=torch.bfloat16,
            device="cuda",
        )
        
        print('======HookedTransformer with sae hooks start=====')    
        start = time.time()
        
        # 创建适配HookedTransformer的hook函数
        def hooked_trans_multi_intervention_hook(activations, hook):
            if not GlobalSAE.use_sae:
                return activations
                
            activations_clone = activations.clone()
            
            if sae.device != activations_clone.device:
                sae.device = activations_clone.device
                sae.to(sae.device)
                
            features = sae.encode(activations_clone)
            reconstructed = sae.decode(features)
            error = activations_clone.to(features.dtype) - reconstructed
            
            for feature_idx, max_activation, strength in zip(feature_idxs, max_activations, strengths):
                features[..., feature_idx] = max_activation * strength
                
            activations_hat = sae.decode(features) + error
            activations_hat = activations_hat.type_as(activations_clone)
            
            return activations_hat
        
        hooks = [
            (f'blocks.{sae.cfg.hook_layer}.hook_mlp_out', hooked_trans_multi_intervention_hook)
        ]
        
        with model.hooks(hooks):
            outputs = model.generate(test_prompts, max_tokens=args.max_output_len, temperature=args.temperature, top_p=args.top_p)
        
        output_tokens = [len(output.split()) for output in outputs]
        hooked_trans_duration = time.time() - start
        output_tokens = np.array(output_tokens)
        print(f"HookedTransformer with sae hooks average output tokens: {output_tokens.mean()}")
        print(f"HookedTransformer with sae hooks cost time: {hooked_trans_duration}s")
        print('======HookedTransformer with sae hooks end=====')
    else:
        print("No SAE model loaded, skipping HookedTransformer test.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    parser.add_argument('--strengths', type=str, default=None)
    args = parser.parse_args()
    main(args)