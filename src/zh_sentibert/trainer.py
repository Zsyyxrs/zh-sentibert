"""
训练器类
包含训练、验证、测试等完整流程
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report, roc_auc_score
)
from typing import Dict, Optional, Tuple
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

class Trainer:
    """模型训练器"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: Optional[DataLoader],
        config,
        logger
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.logger = logger
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # 学习率调度器
        self.scheduler = self._create_scheduler()
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 训练历史
        self.history = defaultdict(list)
        
        # 最佳模型跟踪
        self.best_val_metric = 0
        self.best_epoch = 0
        self.patience_counter = 0
        
        # 混合精度训练（如果可用）
        self.scaler = torch.amp.GradScaler() if config.DEVICE.type == 'cuda' else None
    
    def _create_optimizer(self):
        """创建优化器"""
        # 不同学习率策略
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': self.config.WEIGHT_DECAY
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay) and p.requires_grad],
                'weight_decay': 0.0
            }
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.LEARNING_RATE,
            eps=1e-8
        )
        
        return optimizer
    
    def _create_scheduler(self):
        """创建学习率调度器"""
        total_steps = len(self.train_loader) * self.config.NUM_EPOCHS
        warmup_steps = int(total_steps * self.config.WARMUP_RATIO)
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return scheduler
    
    def train(self) -> str:
        """执行训练"""
        self.logger.info("=" * 50)
        self.logger.info("Starting training...")
        self.logger.info(f"Total epochs: {self.config.NUM_EPOCHS}")
        self.logger.info(f"Batch size: {self.config.BATCH_SIZE}")
        self.logger.info(f"Learning rate: {self.config.LEARNING_RATE}")
        self.logger.info("=" * 50)
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 记录历史
            for key, value in train_metrics.items():
                self.history[f'train_{key}'].append(value)
            for key, value in val_metrics.items():
                self.history[f'val_{key}'].append(value)
            
            # 保存最佳模型
            if val_metrics['accuracy'] > self.best_val_metric:
                self.best_val_metric = val_metrics['accuracy']
                self.best_epoch = epoch
                self.patience_counter = 0
                best_model_path = self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                
            # 定期保存
            if epoch % self.config.SAVE_FREQUENCY == 0 and not self.config.SAVE_BEST_ONLY:
                self.save_checkpoint(epoch, is_best=False)
            
            # 早停检查
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
            
            # 打印进度
            self.print_metrics(epoch, train_metrics, val_metrics)
        
        # 保存训练历史
        self.save_history()
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        self.logger.info(f"Training completed! Best epoch: {self.best_epoch}")
        return best_model_path
    
    def train_epoch(self, epoch: int) -> Dict:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Training Epoch {epoch}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # 将数据移到设备
            input_ids = batch['input_ids'].to(self.config.DEVICE)
            attention_mask = batch['attention_mask'].to(self.config.DEVICE)
            token_type_ids = batch['token_type_ids'].to(self.config.DEVICE)
            labels = batch['labels'].to(self.config.DEVICE)
            
            # 清空梯度
            self.optimizer.zero_grad()
            
            # 混合精度训练
            if self.scaler:
                with torch.amp.autocast(device_type=self.config.DEVICE.type):
                    outputs, loss = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=labels
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs, loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.GRADIENT_CLIP
                )
                self.optimizer.step()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录
            total_loss += loss.item()
            predictions = outputs.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
            
            # 定期记录
            if batch_idx % self.config.LOG_FREQUENCY == 0 and batch_idx > 0:
                current_lr = self.scheduler.get_last_lr()[0]
                self.logger.debug(
                    f"Epoch: {epoch}, Batch: {batch_idx}/{len(self.train_loader)}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.6f}"
                )
        
        # 计算指标
        metrics = self.calculate_metrics(all_predictions, all_labels)
        metrics['loss'] = total_loss / len(self.train_loader)
        
        return metrics
    
    def validate(self, epoch: int) -> Dict:
        """验证"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}", leave=False):
                input_ids = batch['input_ids'].to(self.config.DEVICE)
                attention_mask = batch['attention_mask'].to(self.config.DEVICE)
                token_type_ids = batch['token_type_ids'].to(self.config.DEVICE)
                labels = batch['labels'].to(self.config.DEVICE)
                
                outputs, loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                total_loss += loss.item()
                predictions = outputs.argmax(dim=-1)
                probs = torch.softmax(outputs, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 计算指标
        metrics = self.calculate_metrics(all_predictions, all_labels)
        metrics['loss'] = total_loss / len(self.val_loader)
        
        # 计算AUC（二分类）
        if self.config.NUM_CLASSES == 2:
            metrics['auc'] = roc_auc_score(all_labels, [p[1] for p in all_probs])
        
        return metrics
    
    def evaluate_test(self, checkpoint_path: str) -> Dict:
        """在测试集上评估"""
        if not self.test_loader:
            self.logger.warning("No test loader available")
            return {}
        
        # 加载最佳模型
        checkpoint = torch.load(checkpoint_path, map_location=self.config.DEVICE)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                input_ids = batch['input_ids'].to(self.config.DEVICE)
                attention_mask = batch['attention_mask'].to(self.config.DEVICE)
                token_type_ids = batch['token_type_ids'].to(self.config.DEVICE)
                labels = batch['labels'].to(self.config.DEVICE)
                
                outputs, _ = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids
                )
                
                predictions = outputs.argmax(dim=-1)
                probs = torch.softmax(outputs, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # 计算详细指标
        metrics = self.calculate_metrics(all_predictions, all_labels)
        
        # 生成分类报告
        report = classification_report(
            all_labels, all_predictions,
            target_names=['负向评价', '正向评价'],
            digits=4
        )
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_predictions)
        
        # AUC
        if self.config.NUM_CLASSES == 2:
            metrics['auc'] = roc_auc_score(all_labels, [p[1] for p in all_probs])
        
        # 保存详细结果
        results = {
            'metrics': metrics,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        # 打印结果
        self.logger.info("\n" + "=" * 50)
        self.logger.info("Test Results:")
        self.logger.info("=" * 50)
        self.logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Precision: {metrics['precision']:.4f}")
        self.logger.info(f"Recall: {metrics['recall']:.4f}")
        self.logger.info(f"F1-Score: {metrics['f1']:.4f}")
        if 'auc' in metrics:
            self.logger.info(f"AUC: {metrics['auc']:.4f}")
        self.logger.info("\nClassification Report:")
        self.logger.info("\n" + report)
        
        # 绘制混淆矩阵
        self.plot_confusion_matrix(cm, ['负向评价', '正向评价'])
        
        return results
    
    def calculate_metrics(self, predictions: list, labels: list) -> Dict:
        """计算评估指标"""
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> str:
        """保存模型检查点"""
        if is_best:
            filename = f"best_model_epoch_{epoch}.pth"
        else:
            filename = f"checkpoint_epoch_{epoch}.pth"
        
        checkpoint_path = self.config.CHECKPOINT_DIR / filename
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_metric': self.best_val_metric,
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.config.DEVICE)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['epoch']
    
    def save_history(self):
        """保存训练历史"""
        history_file = self.config.LOG_DIR / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(history_file, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        
        self.logger.info(f"Training history saved: {history_file}")
    
    def print_metrics(self, epoch: int, train_metrics: Dict, val_metrics: Dict):
        """打印指标"""
        self.logger.info(
            f"Epoch {epoch}/{self.config.NUM_EPOCHS} | "
            f"Train Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
            f"F1: {val_metrics['f1']:.4f}"
        )
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss曲线
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy曲线
        axes[0, 1].plot(self.history['train_accuracy'], label='Train')
        axes[0, 1].plot(self.history['val_accuracy'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1曲线
        axes[1, 0].plot(self.history['train_f1'], label='Train')
        axes[1, 0].plot(self.history['val_f1'], label='Validation')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning Rate
        axes[1, 1].plot(range(len(self.history['train_loss'])), 
                       [self.config.LEARNING_RATE] * len(self.history['train_loss']))
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # 保存图片
        plot_file = self.config.RESULT_DIR / f"training_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training curves saved: {plot_file}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: list):
        """绘制混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels
        )
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 保存图片
        cm_file = self.config.RESULT_DIR / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(cm_file, dpi=100, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Confusion matrix saved: {cm_file}")


def evaluate_model(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Dict:
    """独立的模型评估函数"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs, _ = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            predictions = outputs.argmax(dim=-1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='weighted'
    )
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }