#!/bin/bash
set -ex
export VLLM_USE_V1=0

model="/angel/fwk/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
gpu_util=0.8
tensor_parallel_size=1
top_p=0.95
temperature=0.6
max_model_length=22048
max_output_len=20000
feature_idxs=13023,19510,21893,25591,33275,43427,50670,51021,54249,61353
max_activations=10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0
strengths=0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1
sae_release=andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts
sae_id=blocks.19.hook_resid_post

python compare_hookedtrans_vllm.py \
    --model $model \
    --gpu_util $gpu_util \
    --tensor_parallel_size $tensor_parallel_size \
    --top_p $top_p \
    --temperature $temperature \
    --max_model_length $max_model_length \
    --max_output_len $max_output_len \
    --feature_idxs $feature_idxs \
    --max_activations $max_activations \
    --strengths $strengths \
    --sae_release $sae_release \
    --sae_id $sae_id