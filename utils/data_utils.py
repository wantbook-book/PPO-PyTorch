from typing import Iterable, Any, Union
from pathlib import Path
import json
import pandas as pd


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
    if len(data) > 0:
        example = data.iloc[0].to_dict()
        breakpoint()
        with open(output_file.parent / 'example.json', "w", encoding="utf-8") as f:
            json.dump(example, f, ensure_ascii=False, indent=2)
    for item in data:
        with open(output_file.parent / f'{item["id"]}.json', "w", encoding="utf-8") as f:
            # d = {
            #     'problem': 
            # }
            json.dump(item.to_dict(), f, ensure_ascii=False, indent=2)
    

if __name__ == '__main__':
    input_file = "/angel/fwk/code/PPO-PyTorch/dataset/train.parquet"
    output_file = "/angel/fwk/code/PPO-PyTorch/dataset/example.json"
    load_parquet(input_file, output_file)