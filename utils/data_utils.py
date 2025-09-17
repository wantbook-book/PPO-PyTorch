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
                'answer': "\\boxed{" + example['reward_model']['ground_truth'] + "}",
                'data_source': example['data_source'],
                'ability': example['ability'],
                'subject': example['extra_info']['subject'],
                'level': example['extra_info']['level']
            }
            json.dump(d, f, ensure_ascii=False, indent=2)
    idx = 0
    with open(output_file, 'w', encoding="utf-8") as f:
        for _, item in data.iterrows():
            item = item.to_dict()
            d = {
                'idx': idx,
                'problem': item['prompt'][0]['content'].split(think_prompt)[0].strip(),
                'answer': "\\boxed{" + item['reward_model']['ground_truth'] + "}",
                'data_source': example['data_source'],
                'ability': example['ability'],
                'subject': example['extra_info']['subject'],
                'level': example['extra_info']['level']
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

def analysis_scp116k_distribution(input_file):
    """
    统计scp116k数据集中subject、ability、level的数量分布
    
    Args:
        input_file: 输入的jsonl文件路径
    """
    from collections import Counter
    import json
    
    # 初始化计数器
    subject_counter = Counter()
    ability_counter = Counter()
    level_counter = Counter()
    
    total_count = 0
    
    # 读取并统计数据
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    
                    # 统计各字段
                    if 'subject' in data:
                        subject_counter[data['subject']] += 1
                    if 'ability' in data:
                        ability_counter[data['ability']] += 1
                    if 'level' in data:
                        level_counter[data['level']] += 1
                    
                    total_count += 1
                    
                except json.JSONDecodeError:
                    print(f"跳过无效的JSON行: {line[:50]}...")
                    continue
    
    # 打印统计结果
    print(f"=== SCP116K 数据集分布统计 ===")
    print(f"总数据量: {total_count}")
    print()
    
    # Subject分布
    print("📚 Subject分布:")
    print("-" * 40)
    for subject, count in subject_counter.most_common():
        percentage = (count / total_count) * 100
        print(f"{subject:<20}: {count:>6} ({percentage:>5.1f}%)")
    print()
    
    # Ability分布
    print("🎯 Ability分布:")
    print("-" * 40)
    for ability, count in ability_counter.most_common():
        percentage = (count / total_count) * 100
        print(f"{ability:<20}: {count:>6} ({percentage:>5.1f}%)")
    print()
    
    # Level分布
    print("📊 Level分布:")
    print("-" * 40)
    for level, count in sorted(level_counter.items()):
        percentage = (count / total_count) * 100
        print(f"Level {level:<13}: {count:>6} ({percentage:>5.1f}%)")
    print()
    
    # 返回统计结果
    return {
        'total_count': total_count,
        'subject_distribution': dict(subject_counter),
        'ability_distribution': dict(ability_counter),
        'level_distribution': dict(level_counter)
    }


def split_dataset(input_file, train_output_file, test_output_file, test_size=500, random_seed=42):
    """
    将数据集随机分割为训练集和测试集
    
    Args:
        input_file: 输入的jsonl文件路径
        train_output_file: 训练集输出文件路径
        test_output_file: 测试集输出文件路径
        test_size: 测试集大小，默认500条
        random_seed: 随机种子，默认42
    
    Returns:
        dict: 包含分割统计信息的字典
    """
    # 设置随机种子确保可重现性
    random.seed(random_seed)
    
    print(f"🔄 正在读取数据集: {input_file}")
    
    # 读取所有数据
    all_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    all_data.append(data)
                except json.JSONDecodeError:
                    print(f"⚠️  跳过第{line_num}行无效的JSON数据")
                    continue
    
    total_count = len(all_data)
    print(f"📊 总数据量: {total_count}")
    
    # 检查测试集大小是否合理
    if test_size >= total_count:
        raise ValueError(f"测试集大小({test_size})不能大于等于总数据量({total_count})")
    
    # 随机打乱数据
    random.shuffle(all_data)
    
    # 分割数据
    test_data = all_data[:test_size]
    train_data = all_data[test_size:]
    
    # 创建输出目录
    Path(train_output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(test_output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # 保存测试集
    print(f"💾 保存测试集到: {test_output_file}")
    with open(test_output_file, 'w', encoding='utf-8') as f:
        for data in test_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # 保存训练集
    print(f"💾 保存训练集到: {train_output_file}")
    with open(train_output_file, 'w', encoding='utf-8') as f:
        for data in train_data:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    # 统计信息
    stats = {
        'total_count': total_count,
        'train_count': len(train_data),
        'test_count': len(test_data),
        'train_percentage': (len(train_data) / total_count) * 100,
        'test_percentage': (len(test_data) / total_count) * 100,
        'random_seed': random_seed
    }
    
    # 打印分割结果
    print(f"\n✅ 数据集分割完成!")
    print(f"📈 训练集: {stats['train_count']} 条 ({stats['train_percentage']:.1f}%)")
    print(f"📉 测试集: {stats['test_count']} 条 ({stats['test_percentage']:.1f}%)")
    print(f"🎲 随机种子: {random_seed}")
    
    return stats


if __name__ == '__main__':
    # 转换parquet到jsonl
    input_file = Path("/pubshare/fwk/code/sae/PPO-PyTorch/dataset/scp116k/train.parquet")
    output_file = Path("/pubshare/fwk/code/sae/PPO-PyTorch/dataset/scp116k/train.jsonl")
    # load_parquet(input_file, output_file)  # 如果已经转换过，可以注释掉
    
    # 统计scp116k数据分布
    # jsonl_file = "/pubshare/fwk/code/sae/PPO-PyTorch/dataset/scp116k/train.jsonl"
    # print("=" * 60)
    # stats = analysis_scp116k_distribution(jsonl_file)
    
    # 分割数据集为训练集和测试集
    print("\n" + "=" * 60)
    print("🔀 开始分割数据集...")
    jsonl_file = "/pubshare/fwk/code/sae/PPO-PyTorch/dataset/scp116k/train.jsonl"
    train_file = "/pubshare/fwk/code/sae/PPO-PyTorch/dataset/scp116k/train_split.jsonl"
    test_file = "/pubshare/fwk/code/sae/PPO-PyTorch/dataset/scp116k/test_split.jsonl"
    
    split_stats = split_dataset(
        input_file=jsonl_file,
        train_output_file=train_file,
        test_output_file=test_file,
        test_size=500,
        random_seed=42
    )

    # input_file = "/angel/fwk/code/SAE-Reasoning/vllm_sae_evaluation/dataset/gpqa_diamond.jsonl"
    # output_file = '/angel/fwk/code/PPO-PyTorch/dataset/gpqa_diamond/test.jsonl'
    # convert_gpqa_to_jsonl(input_file, output_file)