"""
数据处理工具
包含数据加载、预处理、增强等功能
"""
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from typing import List, Dict, Tuple, Optional
import random
import jieba
from collections import Counter
import re

class ImprovedDataset(Dataset):
    """改进的数据集类"""
    
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 128,
        augment: bool = False,
        augment_prob: float = 0.1
    ):
        """
        Args:
            texts: 文本列表
            labels: 标签列表
            tokenizer: 分词器
            max_length: 最大序列长度
            augment: 是否使用数据增强
            augment_prob: 数据增强概率
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augment = augment
        self.augment_prob = augment_prob
        
        # 构建同义词词典（简化版）
        self.synonyms = {
            '好': ['不错', '优秀', '很好', '挺好'],
            '差': ['糟糕', '不好', '很差', '糟'],
            '喜欢': ['爱', '中意', '满意'],
            '讨厌': ['厌恶', '不喜欢', '反感'],
            '漂亮': ['美丽', '好看', '美观'],
            '难看': ['丑陋', '不好看', '丑']
        }
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 数据增强
        if self.augment and random.random() < self.augment_prob:
            text = self._augment_text(text)
        
        # 编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
    def _augment_text(self, text: str) -> str:
        """文本增强：同义词替换"""
        words = list(jieba.cut(text))
        augmented_words = []
        
        for word in words:
            if word in self.synonyms and random.random() < 0.3:
                # 30%概率替换同义词
                augmented_words.append(random.choice(self.synonyms[word]))
            else:
                augmented_words.append(word)
        
        return ''.join(augmented_words)


class DataProcessor:
    """数据处理器"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """文本清洗"""
        # 移除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        # 移除URLs
        text = re.sub(r'http\S+|www.\S+', '', text)
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text)
        # 移除特殊字符（保留中文、英文、数字和基本标点）
        text = re.sub(r'[^\w\s\u4e00-\u9fa5。，！？、；：""''（）《》【】]', '', text)
        return text.strip()
    
    @staticmethod
    def balance_dataset(
        texts: List[str],
        labels: List[int],
        strategy: str = 'undersample'
    ) -> Tuple[List[str], List[int]]:
        """
        平衡数据集
        
        Args:
            texts: 文本列表
            labels: 标签列表
            strategy: 平衡策略 ('undersample', 'oversample', 'smote')
        """
        label_counts = Counter(labels)
        
        if strategy == 'undersample':
            # 欠采样：减少多数类
            min_count = min(label_counts.values())
            balanced_texts = []
            balanced_labels = []
            
            for label in label_counts:
                label_indices = [i for i, l in enumerate(labels) if l == label]
                sampled_indices = random.sample(label_indices, min_count)
                balanced_texts.extend([texts[i] for i in sampled_indices])
                balanced_labels.extend([label] * min_count)
            
        elif strategy == 'oversample':
            # 过采样：增加少数类
            max_count = max(label_counts.values())
            balanced_texts = []
            balanced_labels = []
            
            for label in label_counts:
                label_texts = [texts[i] for i, l in enumerate(labels) if l == label]
                current_count = len(label_texts)
                
                if current_count < max_count:
                    # 随机重复采样
                    additional_samples = random.choices(
                        label_texts,
                        k=max_count - current_count
                    )
                    label_texts.extend(additional_samples)
                
                balanced_texts.extend(label_texts)
                balanced_labels.extend([label] * max_count)
        
        else:
            balanced_texts = texts
            balanced_labels = labels
        
        # 打乱数据
        combined = list(zip(balanced_texts, balanced_labels))
        random.shuffle(combined)
        balanced_texts, balanced_labels = zip(*combined)
        
        return list(balanced_texts), list(balanced_labels)
    
    @staticmethod
    def create_weighted_sampler(labels: List[int]) -> WeightedRandomSampler:
        """创建加权采样器（用于不平衡数据）"""
        label_counts = Counter(labels)
        weights = {label: 1.0 / count for label, count in label_counts.items()}
        sample_weights = [weights[label] for label in labels]
        
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )


def create_data_loaders(
    dataset,
    tokenizer,
    config,
    use_kfold: bool = False,
    n_splits: int = 5
) -> Tuple:
    """
    创建数据加载器
    
    Args:
        dataset: 数据集
        tokenizer: 分词器
        config: 配置对象
        use_kfold: 是否使用K折交叉验证
        n_splits: K折的折数
    """
    processor = DataProcessor()
    
    if use_kfold:
        # K折交叉验证
        texts = dataset['train']['text']
        labels = dataset['train']['label']
        
        # 清洗文本
        texts = [processor.clean_text(text) for text in texts]
        
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_SEED)
        fold_loaders = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(texts, labels)):
            train_texts = [texts[i] for i in train_idx]
            train_labels = [labels[i] for i in train_idx]
            val_texts = [texts[i] for i in val_idx]
            val_labels = [labels[i] for i in val_idx]
            
            # 平衡训练数据
            train_texts, train_labels = processor.balance_dataset(
                train_texts, train_labels, strategy='oversample'
            )
            
            # 创建数据集
            train_dataset = ImprovedDataset(
                train_texts, train_labels, tokenizer,
                config.MAX_LENGTH, augment=True
            )
            val_dataset = ImprovedDataset(
                val_texts, val_labels, tokenizer,
                config.MAX_LENGTH, augment=False
            )
            
            # 创建加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=config.EVAL_BATCH_SIZE,
                shuffle=False,
                num_workers=config.NUM_WORKERS,
                pin_memory=config.PIN_MEMORY
            )
            
            fold_loaders.append((train_loader, val_loader))
        
        return fold_loaders
    
    else:
        # 常规数据划分
        # 获取训练数据
        train_texts = dataset['train']['text']
        train_labels = dataset['train']['label']
        val_texts = dataset['validation']['text']
        val_labels = dataset['validation']['label']
        
        # # 清洗文本
        # train_texts = [processor.clean_text(text) for text in train_texts]
        
        # # 划分训练集和验证集
        # train_texts, val_texts, train_labels, val_labels = train_test_split(
        #     train_texts, train_labels,
        #     test_size=config.VALIDATION_SPLIT,
        #     random_state=config.RANDOM_SEED,
        #     stratify=train_labels
        # )

        
        # 平衡训练数据（可选）
        # train_texts, train_labels = processor.balance_dataset(
        #     train_texts, train_labels, strategy='oversample'
        # )
        
        # 获取测试数据
        test_texts = dataset['test']['text'] if 'test' in dataset else []
        test_labels = dataset['test']['label'] if 'test' in dataset else []
        test_texts = [processor.clean_text(text) for text in test_texts]
        
        # 创建数据集
        train_dataset = ImprovedDataset(
            train_texts, train_labels, tokenizer,
            config.MAX_LENGTH, augment=True, augment_prob=0.1
        )
        val_dataset = ImprovedDataset(
            val_texts, val_labels, tokenizer,
            config.MAX_LENGTH, augment=False
        )
        test_dataset = ImprovedDataset(
            test_texts, test_labels, tokenizer,
            config.MAX_LENGTH, augment=False
        ) if test_texts else None
        
        # 创建数据加载器
        # 使用加权采样器处理不平衡数据
        train_sampler = processor.create_weighted_sampler(train_labels)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            sampler=train_sampler,  # 使用加权采样器
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY
        ) if test_dataset else None
        
        return train_loader, val_loader, test_loader


class DataAugmenter:
    """高级数据增强类"""
    
    def __init__(self):
        self.methods = [
            self.synonym_replacement,
            self.random_insertion,
            self.random_swap,
            self.random_deletion,
            self.back_translation_simulation
        ]
    
    def augment(self, text: str, n_aug: int = 1) -> List[str]:
        """生成增强文本"""
        augmented_texts = []
        
        for _ in range(n_aug):
            method = random.choice(self.methods)
            augmented_text = method(text)
            augmented_texts.append(augmented_text)
        
        return augmented_texts
    
    def synonym_replacement(self, text: str, n: int = 2) -> str:
        """同义词替换"""
        words = list(jieba.cut(text))
        new_words = words.copy()
        random_word_list = list(set([word for word in words if len(word) > 1]))
        random.shuffle(random_word_list)
        
        num_replaced = 0
        for random_word in random_word_list:
            # 这里应该使用同义词词典
            # 简化处理：随机替换为相似的词
            if num_replaced >= n:
                break
            num_replaced += 1
        
        return ''.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """随机插入"""
        words = list(jieba.cut(text))
        for _ in range(n):
            new_word = self._get_random_word(words)
            idx = random.randint(0, len(words))
            words.insert(idx, new_word)
        return ''.join(words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """随机交换"""
        words = list(jieba.cut(text))
        for _ in range(n):
            if len(words) < 2:
                break
            idx1 = random.randint(0, len(words) - 1)
            idx2 = random.randint(0, len(words) - 1)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        return ''.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """随机删除"""
        words = list(jieba.cut(text))
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        if len(new_words) == 0:
            return words[random.randint(0, len(words) - 1)]
        
        return ''.join(new_words)
    
    def back_translation_simulation(self, text: str) -> str:
        """模拟回译（简化版）"""
        # 实际应用中应该使用翻译API
        # 这里只是简单的词序调整
        words = list(jieba.cut(text))
        if len(words) > 3:
            # 随机调整部分词序
            start = random.randint(0, len(words) - 3)
            words[start:start + 3] = reversed(words[start:start + 3])
        return ''.join(words)
    
    def _get_random_word(self, words: List[str]) -> str:
        """获取随机词"""
        return random.choice(words) if words else "的"