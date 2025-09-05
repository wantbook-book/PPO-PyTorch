from typing import Iterable, Any, Union
from pathlib import Path
import json
import pandas as pd
import random
random.seed(42)
def load_txt(file: Union[str, Path]) -> str:
    with open(file, "r", encoding="utf-8") as f:
        return f.read()

def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                yield json.loads(line)
            except:
                print("Error in loading:", line)
                exit()

def load_parquet(file: Union[str, Path], output_file) -> Iterable[Any]:
    data =  pd.read_parquet(file)
    # 保存出一个example json
    # 保存第一行数据作为示例
    think_prompt = "Let's think step by step and output the final answer"
    if len(data) > 0:
        example = data.iloc[0].to_dict()
        with open(output_file.parent / 'example.json', "w", encoding="utf-8") as f:
            d = {
                'idx': 0,
                'problem': example['prompt'][0]['content'].split(think_prompt)[0].strip(),
                'answer': "\\boxed{" + example['reward_model']['ground_truth'] + "}"
            }
            json.dump(d, f, ensure_ascii=False, indent=2)
    idx = 0
    with open(output_file, 'w', encoding="utf-8") as f:
        for _, item in data.iterrows():
            item = item.to_dict()
            d = {
                'idx': idx,
                'problem': item['prompt'][0]['content'].split(think_prompt)[0].strip(),
                'answer': "\\boxed{" + item['reward_model']['ground_truth'] + "}"
            }
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
            idx += 1
    exit()

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]

def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    
    return prompt

def convert_gpqa_to_jsonl(input_file, output_file):
    data_list = []
    with open(input_file, 'r') as f:
        for line in f:
            data_list.append(json.loads(line))
    with open(output_file, 'w') as f:
        idx = 0
        for data in data_list:
            options = data['options']
            random.shuffle(options)
            item = {
                'idx': idx,
                'problem': format_cot_example(data, including_answer=False),
                'answer': "\\boxed{" + choices[options.index(data["answer"])] + "}",
                'choices_num': len(options)
            }
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            idx += 1
    exit()

if __name__ == '__main__':
    # input_file = Path("/angel/fwk/code/PPO-PyTorch/dataset/train.parquet")
    # output_file = Path("/angel/fwk/code/PPO-PyTorch/dataset/train.jsonl")
    # load_parquet(input_file, output_file)

    input_file = "/angel/fwk/code/SAE-Reasoning/vllm_sae_evaluation/dataset/gpqa_diamond.jsonl"
    output_file = '/angel/fwk/code/PPO-PyTorch/dataset/gpqa_diamond/test.jsonl'
    convert_gpqa_to_jsonl(input_file, output_file)