from datasets import load_dataset
# config.py
import os
from pathlib import Path
# from dotenv import load_dotenv  # 需要: pip install python-dotenv
# Load model directly
from transformers import AutoTokenizer, AutoModelForMaskedLM

# tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")
# model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-chinese")
# print(model)

# 示例：下载中文情感分析数据集
# dataset = load_dataset("lansinuote/ChnSentiCorp")
dataset = load_dataset(
    "lansinuote/ChnSentiCorp",
    cache_dir="/root/.cache/huggingface/datasets",
    download_mode="reuse_dataset_if_exists",
    verification_mode="no_checks"
    )

# 数据默认缓存到 ~/.cache/huggingface/datasets 下
# 如果要保存为本地文件：
# dataset.save_to_disk("./ChnSentiCorp")

# 加载 .env 文件
# load_dotenv()

# print(os.environ.get('MODEL_PATH') + '/bert-base-chinese')
# print(os.environ.get('DATA_PATH'))
# texts = dataset['train']

# 'test', 'train', 'validation'
# train_texts = dataset['validation']['text']
# train_labels = dataset['validation']['label']
# print(train_texts[:2])
# print(train_labels[:2])

print(len(dataset['test']))
print(len(dataset['validation']))
print(len(dataset['train']))