#!/bin/bash
set -ex
export VLLM_USE_V1=0

clip_range=0.2
clip_range_low=0.2
clip_range_high=0.2
clip_ratio_c=3.0
tensor_parallel_size=1

top_p=0.95
temperature=0.6
max_model_length=22048
max_input_len=2048
max_output_len=20000
gpu_util=0.8
lr=1e-4
kl_coef=0
vocab_size=128256
logprobs=20

# train_data_path="/angel/fwk/code/PPO-PyTorch/dataset/scp116k/train.jsonl"
# test_data_path="/angel/fwk/code/PPO-PyTorch/dataset/gpqa_diamond/test.jsonl"
# train_prompt_path="/angel/fwk/code/PPO-PyTorch/prompts/multichoice.txt"
# test_prompt_path="/angel/fwk/code/PPO-PyTorch/prompts/multichoice.txt"
proj_path="/pubshare/fwk/code/sae/PPO-PyTorch"
train_data_path="${proj_path}/dataset/scp116k/train.jsonl"
test_data_path="${proj_path}/dataset/gpqa_diamond/test.jsonl"
train_prompt_path="${proj_path}/prompts/multichoice.txt"
test_prompt_path="${proj_path}/prompts/multichoice.txt"


# model="/angel/fwk/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# num_instances=6
num_instances=2
sae_release=andreuka18/deepseek-r1-distill-llama-8b-lmsys-openthoughts
sae_id=blocks.19.hook_resid_post
save_dir="./checkpoints/strength_predictor_DeepSeek-R1-Distill-Llama-8B"
save_interval=50
eval_interval=10
num_epochs=2
batch_size=4
feature_idxs=13023,19510,21893,25591,33275,43427,50670,51021,54249,61353
# max_activations=20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0,20.0
max_activations=10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0,10.0

loss_agg_mode="token-mean"
n_samples_per_prompt=16
intensity_std=0.1

python train_sae_strength.py \
    --train_data_path $train_data_path \
    --test_data_path $test_data_path \
    --train_prompt_path $train_prompt_path \
    --test_prompt_path $test_prompt_path \
    --batch_size $batch_size \
    --model $model \
    --tensor_parallel_size $tensor_parallel_size \
    --gpu_util $gpu_util \
    --top_p $top_p \
    --temperature $temperature \
    --max_model_length $max_model_length \
    --max_input_len $max_input_len \
    --max_output_len $max_output_len \
    --n_samples_per_prompt $n_samples_per_prompt \
    --sae_release $sae_release \
    --sae_id $sae_id \
    --feature_idxs $feature_idxs \
    --max_activations $max_activations \
    --clip_range $clip_range \
    --clip_range_low $clip_range_low \
    --clip_range_high $clip_range_high \
    --clip_ratio_c $clip_ratio_c \
    --loss_agg_mode $loss_agg_mode \
    --lr $lr \
    --num_epochs $num_epochs \
    --save_dir $save_dir \
    --save_interval $save_interval \
    --kl_coef $kl_coef \
    --vocab_size $vocab_size \
    --logprobs $logprobs \
    --eval_interval $eval_interval \
    --num_instances $num_instances \
    --intensity_std $intensity_std