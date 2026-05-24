"""
推理预测模块
支持单条/批量预测、置信度分析等
"""
import torch
import torch.nn as nn
from typing import List, Union, Dict, Optional
import numpy as np
from transformers import BertTokenizer
from pathlib import Path
import json
from tqdm import tqdm
import time

class SentimentPredictor:
    """情感预测器"""
    
    def __init__(
        self,
        model_path: str,
        tokenizer: BertTokenizer,
        config,
        device: Optional[Union[str, torch.device]] = None
    ):
        """
        Args:
            model_path: 模型检查点路径
            tokenizer: 分词器
            config: 配置对象
            device: 运行设备
        """
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(device) if device else config.DEVICE
        
        # 标签映射
        self.labels = ['负向评价', '正向评价']
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        
        # 加载模型
        self.model = self._load_model(model_path)
        
        # 统计信息
        self.prediction_stats = {
            'total_predictions': 0,
            'total_time': 0,
            'confidence_distribution': []
        }
    
    def _load_model(self, model_path: str) -> nn.Module:
        """加载模型"""
        from .model import ImprovedModel
        
        # 初始化模型
        model = ImprovedModel(
            model_name=self.config.MODEL_PATH,
            num_classes=self.config.NUM_CLASSES,
            dropout_rate=0  # 推理时不使用dropout
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded from {model_path}")
        return model
    
    @torch.no_grad()
    def predict(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        return_probs: bool = False,
        threshold: float = 0.5
    ) -> Union[Dict, List[Dict]]:
        """
        预测文本情感
        
        Args:
            texts: 单条文本或文本列表
            batch_size: 批处理大小
            return_probs: 是否返回概率分布
            threshold: 预测阈值（用于不确定性检测）
            
        Returns:
            预测结果字典或字典列表
        """
        # 统一处理为列表
        is_single = isinstance(texts, str)
        if is_single:
            texts = [texts]
        
        # 开始计时
        start_time = time.time()
        
        # 预测结果
        results = []
        
        # 批处理预测
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 编码
            encodings = self.tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=self.config.MAX_LENGTH,
                return_tensors='pt'
            )
            
            # 移到设备
            input_ids = encodings['input_ids'].to(self.device)
            attention_mask = encodings['attention_mask'].to(self.device)
            token_type_ids = encodings['token_type_ids'].to(self.device)
            
            # 前向传播
            outputs, _ = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 计算概率
            probs = torch.softmax(outputs, dim=-1)
            predictions = outputs.argmax(dim=-1)
            
            # 处理结果
            for j, text in enumerate(batch_texts):
                pred_id = predictions[j].item()
                confidence = probs[j, pred_id].item()
                
                result = {
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'label': self.labels[pred_id],
                    'label_id': pred_id,
                    'confidence': confidence
                }
                
                # 不确定性检测
                if confidence < threshold:
                    result['uncertain'] = True
                    result['uncertainty_level'] = 'high' if confidence < 0.6 else 'medium'
                
                # 概率分布
                if return_probs:
                    result['probs'] = {
                        self.labels[k]: probs[j, k].item()
                        for k in range(len(self.labels))
                    }
                
                results.append(result)
                
                # 更新统计
                self.prediction_stats['confidence_distribution'].append(confidence)
        
        # 更新统计信息
        elapsed_time = time.time() - start_time
        self.prediction_stats['total_predictions'] += len(texts)
        self.prediction_stats['total_time'] += elapsed_time
        
        # 返回结果
        return results[0] if is_single else results
    
    def predict_file(
        self,
        file_path: str,
        output_path: Optional[str] = None,
        batch_size: int = 32
    ) -> List[Dict]:
        """
        预测文件中的文本
        
        Args:
            file_path: 输入文件路径（每行一个文本）
            output_path: 输出文件路径
            batch_size: 批处理大小
            
        Returns:
            预测结果列表
        """
        # 读取文本
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(texts)} texts from {file_path}")
        
        # 批量预测
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch = texts[i:i + batch_size]
            batch_results = self.predict(batch, batch_size=batch_size, return_probs=True)
            results.extend(batch_results)
        
        # 保存结果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Results saved to {output_path}")
        
        # 打印统计
        self.print_statistics(results)
        
        return results
    
    def analyze_uncertainty(
        self,
        texts: List[str],
        n_samples: int = 10,
        temperature: float = 1.0
    ) -> List[Dict]:
        """
        使用蒙特卡洛dropout分析预测不确定性
        
        Args:
            texts: 文本列表
            n_samples: 采样次数
            temperature: 温度参数
            
        Returns:
            不确定性分析结果
        """
        # 临时启用dropout
        self.model.train()
        
        results = []
        
        for text in tqdm(texts, desc="Uncertainty Analysis"):
            # 多次采样
            all_probs = []
            
            for _ in range(n_samples):
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=self.config.MAX_LENGTH,
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                token_type_ids = encoding['token_type_ids'].to(self.device)
                
                with torch.no_grad():
                    outputs, _ = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    
                    probs = torch.softmax(outputs / temperature, dim=-1)
                    all_probs.append(probs.cpu().numpy())
            
            # 计算统计量
            all_probs = np.array(all_probs).squeeze()
            mean_probs = np.mean(all_probs, axis=0)
            std_probs = np.std(all_probs, axis=0)
            
            # 预测熵
            entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-8))
            
            # 最终预测
            pred_id = np.argmax(mean_probs)
            
            result = {
                'text': text[:100] + '...' if len(text) > 100 else text,
                'prediction': self.labels[pred_id],
                'mean_confidence': mean_probs[pred_id],
                'std_confidence': std_probs[pred_id],
                'entropy': entropy,
                'mean_probs': {self.labels[i]: mean_probs[i] for i in range(len(self.labels))},
                'std_probs': {self.labels[i]: std_probs[i] for i in range(len(self.labels))}
            }
            
            results.append(result)
        
        # 恢复评估模式
        self.model.eval()
        
        return results
    
    def get_attention_weights(self, text: str) -> Dict:
        """
        获取注意力权重（用于可解释性）
        
        Args:
            text: 输入文本
            
        Returns:
            注意力权重和token信息
        """
        # 编码
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH,
            return_tensors='pt',
            return_offsets_mapping=True
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        token_type_ids = encoding['token_type_ids'].to(self.device)
        
        # 获取token列表
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        
        # 如果模型支持返回注意力权重
        self.model.bert.config.output_attentions = True
        
        with torch.no_grad():
            outputs = self.model.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=True
            )
            
            # 获取最后一层的注意力权重
            attentions = outputs.attentions[-1]  # [batch, heads, seq_len, seq_len]
            
            # 平均所有注意力头
            avg_attention = attentions.mean(dim=1)[0]  # [seq_len, seq_len]
            
            # 获取[CLS] token的注意力
            cls_attention = avg_attention[0].cpu().numpy()
        
        # 恢复设置
        self.model.bert.config.output_attentions = False
        
        return {
            'tokens': tokens,
            'attention_weights': cls_attention.tolist(),
            'token_importance': list(zip(tokens, cls_attention.tolist()))
        }
    
    def batch_similarity(
        self,
        texts: List[str],
        reference_text: str,
        metric: str = 'cosine'
    ) -> List[float]:
        """
        计算文本与参考文本的相似度
        
        Args:
            texts: 文本列表
            reference_text: 参考文本
            metric: 相似度度量方式
            
        Returns:
            相似度列表
        """
        # 获取参考文本嵌入
        ref_encoding = self.tokenizer(
            reference_text,
            truncation=True,
            padding=True,
            max_length=self.config.MAX_LENGTH,
            return_tensors='pt'
        )
        
        ref_input_ids = ref_encoding['input_ids'].to(self.device)
        ref_attention_mask = ref_encoding['attention_mask'].to(self.device)
        ref_token_type_ids = ref_encoding['token_type_ids'].to(self.device)
        
        ref_embedding = self.model.get_embeddings(
            ref_input_ids, ref_attention_mask, ref_token_type_ids
        )
        
        similarities = []
        
        for text in texts:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=self.config.MAX_LENGTH,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            token_type_ids = encoding['token_type_ids'].to(self.device)
            
            embedding = self.model.get_embeddings(
                input_ids, attention_mask, token_type_ids
            )
            
            if metric == 'cosine':
                # 余弦相似度
                similarity = torch.cosine_similarity(
                    ref_embedding, embedding, dim=-1
                ).item()
            elif metric == 'euclidean':
                # 欧氏距离
                similarity = -torch.norm(
                    ref_embedding - embedding, p=2, dim=-1
                ).item()
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            similarities.append(similarity)
        
        return similarities
    
    def print_statistics(self, results: Optional[List[Dict]] = None):
        """打印预测统计信息"""
        print("\n" + "=" * 50)
        print("Prediction Statistics")
        print("=" * 50)
        
        if results:
            # 标签分布
            label_counts = {}
            for r in results:
                label = r['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            
            print("\nLabel Distribution:")
            for label, count in label_counts.items():
                percentage = count / len(results) * 100
                print(f"  {label}: {count} ({percentage:.2f}%)")
            
            # 置信度统计
            confidences = [r['confidence'] for r in results]
            print(f"\nConfidence Statistics:")
            print(f"  Mean: {np.mean(confidences):.4f}")
            print(f"  Std: {np.std(confidences):.4f}")
            print(f"  Min: {np.min(confidences):.4f}")
            print(f"  Max: {np.max(confidences):.4f}")
            
            # 不确定样本
            uncertain = [r for r in results if r.get('uncertain', False)]
            if uncertain:
                print(f"\nUncertain Predictions: {len(uncertain)} ({len(uncertain)/len(results)*100:.2f}%)")
        
        # 总体统计
        if self.prediction_stats['total_predictions'] > 0:
            avg_time = self.prediction_stats['total_time'] / self.prediction_stats['total_predictions']
            print(f"\nOverall Performance:")
            print(f"  Total Predictions: {self.prediction_stats['total_predictions']}")
            print(f"  Total Time: {self.prediction_stats['total_time']:.2f}s")
            print(f"  Average Time per Prediction: {avg_time*1000:.2f}ms")
        
        print("=" * 50)


class InteractivePredictorr:
    """交互式预测器"""
    
    def __init__(self, predictor: SentimentPredictor):
        self.predictor = predictor
        self.history = []
    
    def run(self):
        """运行交互式预测"""
        print("\n" + "=" * 50)
        print("Interactive Sentiment Analysis")
        print("=" * 50)
        print("Commands:")
        print("  'quit' - Exit the program")
        print("  'history' - Show prediction history")
        print("  'stats' - Show statistics")
        print("  'explain' - Explain last prediction")
        print("=" * 50 + "\n")
        
        while True:
            text = input("\nEnter text (or command): ").strip()
            
            if not text:
                continue
            
            if text.lower() == 'quit':
                print("Goodbye!")
                break
            
            elif text.lower() == 'history':
                self.show_history()
            
            elif text.lower() == 'stats':
                self.predictor.print_statistics(self.history)
            
            elif text.lower() == 'explain' and self.history:
                self.explain_last_prediction()
            
            else:
                # 预测
                result = self.predictor.predict(text, return_probs=True)
                self.history.append(result)
                
                # 显示结果
                self.display_result(result)
    
    def display_result(self, result: Dict):
        """显示预测结果"""
        print("\n" + "-" * 30)
        print(f"预测结果: {result['label']}")
        print(f"置信度: {result['confidence']:.4f}")
        
        if 'probs' in result:
            print("\n概率分布:")
            for label, prob in result['probs'].items():
                bar = '█' * int(prob * 20)
                print(f"  {label:8s}: {bar:20s} {prob:.4f}")
        
        if result.get('uncertain', False):
            print(f"\n⚠️ 警告: 预测不确定性较高 ({result.get('uncertainty_level', 'unknown')})")
        
        print("-" * 30)
    
    def show_history(self):
        """显示历史记录"""
        if not self.history:
            print("No prediction history yet.")
            return
        
        print("\n" + "=" * 50)
        print("Prediction History")
        print("=" * 50)
        
        for i, result in enumerate(self.history[-10:], 1):  # 显示最近10条
            text = result['text'][:50] + '...' if len(result['text']) > 50 else result['text']
            print(f"{i}. {text}")
            print(f"   → {result['label']} ({result['confidence']:.4f})")
    
    def explain_last_prediction(self):
        """解释最后一次预测"""
        if not self.history:
            print("No predictions to explain.")
            return
        
        last_result = self.history[-1]
        
        # 获取注意力权重
        attention_info = self.predictor.get_attention_weights(last_result['text'])
        
        print("\n" + "=" * 50)
        print("Prediction Explanation")
        print("=" * 50)
        
        # 显示重要词汇
        print("\nMost Important Tokens:")
        important_tokens = sorted(
            attention_info['token_importance'],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        for token, weight in important_tokens:
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                bar = '█' * int(weight * 50)
                print(f"  {token:10s}: {bar:25s} {weight:.4f}")