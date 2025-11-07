"""
主训练脚本
使用方法：python main_train.py --epochs 10 --batch_size 32
"""
import argparse
import json
import logging
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertTokenizer, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import warnings
from datasets import load_dataset

warnings.filterwarnings('ignore')

from config import Config
from model import ImprovedModel
from data_utils import create_data_loaders, ImprovedDataset
from trainer import Trainer

def setup_logging(config):
    """设置日志"""
    log_file = config.LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='中文情感分析模型训练')
    
    # 基本参数
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'predict'],
                       help='运行模式: train/eval/predict')
    
    # 数据参数
    # parser.add_argument('--data_path', type=str, default=None,
    #                    help='数据路径（覆盖配置文件）')
    parser.add_argument('--max_length', type=int, default=None,
                       help='最大序列长度')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default=None,
                       help='预训练模型名称或路径')
    parser.add_argument('--dropout', type=float, default=None,
                       help='Dropout率')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    parser.add_argument('--warmup_ratio', type=float, default=None,
                       help='预热比例')
    
    # 其他参数
    parser.add_argument('--seed', type=int, default=None,
                       help='随机种子')
    parser.add_argument('--device', type=str, default=None,
                       choices=['cpu', 'cuda'],
                       help='运行设备')
    parser.add_argument('--checkpoint', type=str, default=None,
                   help='检查点文件路径（用于继续训练或评估）')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                   help='检查点保存目录')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--use_wandb', action='store_true',
                       help='使用wandb记录')
    
    # 推理参数
    parser.add_argument('--input_text', type=str, default=None,
                       help='待预测的文本（predict模式）')
    parser.add_argument('--input_file', type=str, default=None,
                       help='待预测的文本文件（predict模式）')
    
    return parser.parse_args()

def update_config(config, args):
    """根据命令行参数更新配置"""
    # if args.data_path:
    #     config.DATA_PATH = Path(args.data_path)
    if args.max_length:
        config.MAX_LENGTH = args.max_length
    if args.model_name:
        config.MODEL_PATH = args.model_name
    if args.dropout:
        config.DROPOUT_RATE = args.dropout
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.warmup_ratio:
        config.WARMUP_RATIO = args.warmup_ratio
    if args.seed:
        config.RANDOM_SEED = args.seed
    if args.device:
        config.DEVICE = torch.device(args.device)
    if args.checkpoint_dir:
        config.CHECKPOINT_DIR = Path(args.checkpoint_dir)
        config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if args.output_dir:
        config.RESULT_DIR = Path(args.output_dir)
        config.RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if args.use_wandb:
        config.USE_WANDB = True
    
    return config

def load_data(config, tokenizer, logger):
    """加载数据"""
    logger.info("Loading data...")
    
    # try:
    #     # 尝试从本地加载数据
    #     from datasets import load_from_disk
    #     dataset = load_from_disk(str(config.DATA_PATH))
    #     dataset = load_dataset("lansinuote/ChnSentiCorp")
    #     logger.info(f"Data loaded from {config.DATA_PATH}")
    # except Exception as e:
    #     logger.error(f"Failed to load data: {e}")
    #     logger.info("Attempting to load from online source...")
    #     dataset = load_dataset("seamew/ChnSentiCorp")
    #     # 保存到本地
    #     dataset.save_to_disk(str(config.DATA_PATH))
    #     logger.info(f"Data saved to {config.DATA_PATH}")
    logger.info("Attempting to load from online source...")
    # dataset = load_dataset("lansinuote/ChnSentiCorp")
    dataset = load_dataset(
    "lansinuote/ChnSentiCorp",
    cache_dir="/root/.cache/huggingface/datasets",
    download_mode="reuse_dataset_if_exists",
    verification_mode="no_checks"
    )
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        dataset, tokenizer, config
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    return train_loader, val_loader, test_loader

def train_mode(config, logger):
    """训练模式"""
    logger.info("Starting training mode...")
    
    # 初始化tokenizer
    logger.info(f"Loading tokenizer from {config.MODEL_PATH}")
    # tokenizer = BertTokenizer.from_pretrained(config.MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    
    # 加载数据
    train_loader, val_loader, test_loader = load_data(config, tokenizer, logger)
    
    # 初始化模型
    logger.info("Initializing model...")
    model = ImprovedModel(
        model_name=config.MODEL_PATH,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE,
        unfreeze_layers=config.UNFREEZE_LAYERS
    )
    model.to(config.DEVICE)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # 初始化训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        logger=logger
    )
    
    # 开始训练
    logger.info("Starting training...")
    best_model_path = trainer.train()
    
    # 在测试集上评估
    logger.info("Evaluating on test set...")
    test_results = trainer.evaluate_test(best_model_path)
    
    # 保存结果
    results_file = config.RESULT_DIR / f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")
    logger.info("Training completed!")
    
    return best_model_path

def eval_mode(config, logger, checkpoint_path):
    """评估模式"""
    logger.info("Starting evaluation mode...")
    
    if not checkpoint_path:
        logger.error("Checkpoint path is required for evaluation mode")
        return
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    
    # 加载数据
    _, _, test_loader = load_data(config, tokenizer, logger)
    
    # 加载模型
    logger.info(f"Loading model from {checkpoint_path}")
    model = ImprovedModel(
        model_name=config.MODEL_PATH,
        num_classes=config.NUM_CLASSES,
        dropout_rate=config.DROPOUT_RATE
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(config.DEVICE)
    model.eval()
    
    # 评估
    from trainer import evaluate_model
    results = evaluate_model(model, test_loader, config.DEVICE)
    
    # 打印结果
    logger.info("\nEvaluation Results:")
    logger.info(f"Accuracy: {results['accuracy']:.4f}")
    logger.info(f"Precision: {results['precision']:.4f}")
    logger.info(f"Recall: {results['recall']:.4f}")
    logger.info(f"F1-Score: {results['f1']:.4f}")
    
    return results

def predict_mode(config, logger, checkpoint_path, input_text=None, input_file=None):
    """预测模式"""
    logger.info("Starting prediction mode...")
    
    if not checkpoint_path:
        # 尝试找到最新的最佳模型
        checkpoint_files = list(config.CHECKPOINT_DIR.glob("best_model_*.pth"))
        if not checkpoint_files:
            logger.error("No checkpoint found. Please train a model first.")
            return
        checkpoint_path = max(checkpoint_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"Using latest checkpoint: {checkpoint_path}")
    
    # 初始化predictor
    from inference import SentimentPredictor
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_PATH)
    predictor = SentimentPredictor(
        model_path=str(checkpoint_path),
        tokenizer=tokenizer,
        config=config,
        device=config.DEVICE
    )
    
    # 准备预测文本
    texts = []
    if input_text:
        texts = [input_text]
    elif input_file:
        with open(input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # 交互式输入
        logger.info("Enter text for prediction (type 'quit' to exit):")
        while True:
            text = input("\n> ")
            if text.lower() == 'quit':
                break
            
            # 预测
            result = predictor.predict(text, return_probs=True)
            
            # 打印结果
            print(f"\n预测结果: {result['label']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"概率分布:")
            for label, prob in result['probs'].items():
                print(f"  - {label}: {prob:.4f}")
        return
    
    # 批量预测
    if texts:
        logger.info(f"Predicting {len(texts)} texts...")
        results = predictor.predict(texts, return_probs=True)
        
        # 确保results是列表
        if not isinstance(results, list):
            results = [results]
        
        # 打印结果
        for i, result in enumerate(results):
            print(f"\n文本 {i+1}: {result['text'][:50]}...")
            print(f"预测: {result['label']} (置信度: {result['confidence']:.4f})")
            if 'probs' in result:
                print(f"概率: {result['probs']}")
        
        # 保存结果
        output_file = config.RESULT_DIR / f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Predictions saved to {output_file}")

def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    # 更新配置
    config = update_config(Config, args)
    
    # 设置日志
    logger = setup_logging(config)
    
    # 打印配置
    if args.mode == 'train':
        config.print_config()
    
    # 设置随机种子
    set_seed(config.RANDOM_SEED)
    logger.info(f"Random seed set to {config.RANDOM_SEED}")
    
    # 设置Weights & Biases
    if config.USE_WANDB:
        import wandb
        wandb.init(
            project=config.WANDB_PROJECT,
            config=config.to_dict(),
            name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    
    try:
        # 根据模式执行
        if args.mode == 'train':
            train_mode(config, logger)
        elif args.mode == 'eval':
            eval_mode(config, logger, args.checkpoint)
        elif args.mode == 'predict':
            predict_mode(config, logger, args.checkpoint, args.input_text, args.input_file)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        if config.USE_WANDB:
            wandb.finish()

if __name__ == "__main__":
    main()