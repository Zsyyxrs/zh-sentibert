"""
项目配置文件
"""
import os
import torch
from pathlib import Path
from datasets import load_dataset


class Config:
    """项目配置类"""
    
    # ============ 路径配置 ============
    BASE_DIR = Path(__file__).parent
    
    # 模型路径 - 可通过环境变量设置，否则使用默认值
    MODEL_PATH = 'google-bert/bert-base-chinese'
    
    # 数据路径
    # DATA_PATH = BASE_DIR / 'data' / 'ChnSentiCorp'
    RAW_DATA_PATH = BASE_DIR / 'data' / 'raw'
    PROCESSED_DATA_PATH = BASE_DIR / 'data' / 'processed'
    
    # 输出路径
    CHECKPOINT_DIR = BASE_DIR / 'checkpoints'
    LOG_DIR = BASE_DIR / 'logs'
    RESULT_DIR = BASE_DIR / 'results'
    
    # 创建必要的目录
    for dir_path in [CHECKPOINT_DIR, LOG_DIR, RESULT_DIR, PROCESSED_DATA_PATH]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # ============ 模型配置 ============
    NUM_CLASSES = 2  # 分类数量
    DROPOUT_RATE = 0.3  # Dropout率
    HIDDEN_SIZE = 768  # BERT隐藏层大小
    UNFREEZE_LAYERS = 2  # 解冻BERT最后几层
    
    # ============ 训练配置 ============
    BATCH_SIZE = 32  # 批次大小
    EVAL_BATCH_SIZE = 64  # 评估批次大小
    MAX_LENGTH = 128  # 最大序列长度
    LEARNING_RATE = 2e-5  # 学习率
    WEIGHT_DECAY = 0.01  # 权重衰减
    NUM_EPOCHS = 10  # 训练轮数
    WARMUP_RATIO = 0.1  # 预热比例
    GRADIENT_CLIP = 1.0  # 梯度裁剪
    
    # ============ 训练策略 ============
    EARLY_STOPPING_PATIENCE = 5  # 早停耐心值
    SAVE_BEST_ONLY = True  # 只保存最佳模型
    SAVE_FREQUENCY = 1  # 保存频率（每n个epoch）
    VALIDATION_SPLIT = 0.1  # 验证集比例
    TEST_SPLIT = 0.1  # 测试集比例
    RANDOM_SEED = 42  # 随机种子
    
    # ============ 设备配置 ============
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_WORKERS = 4 if torch.cuda.is_available() else 0  # 数据加载线程数
    PIN_MEMORY = True if torch.cuda.is_available() else False
    
    # ============ 日志配置 ============
    LOG_LEVEL = 'INFO'  # 日志级别
    LOG_FREQUENCY = 10  # 每n个batch记录一次
    USE_WANDB = False  # 是否使用wandb记录
    WANDB_PROJECT = 'chinese-sentiment-analysis'  # wandb项目名
    
    # ============ 推理配置 ============
    INFERENCE_BATCH_SIZE = 128  # 推理批次大小
    USE_FP16 = False  # 是否使用半精度推理
    
    @classmethod
    def to_dict(cls):
        """将配置转换为字典"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('__') and not callable(value)
        }
    
    @classmethod
    def print_config(cls):
        """打印配置信息"""
        print("\n" + "="*50)
        print("Configuration Settings:")
        print("="*50)
        for key, value in cls.to_dict().items():
            if not key.startswith('__'):
                print(f"{key:25s}: {value}")
        print("="*50 + "\n")