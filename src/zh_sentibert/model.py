"""
改进的BERT模型定义
支持多种微调策略
"""
import torch
import torch.nn as nn
from transformers import AutoModelForMaskedLM, BertConfig
from typing import Optional, Tuple

class ImprovedModel(nn.Module):
    """改进的BERT情感分类模型"""
    
    def __init__(
        self,
        model_name: str = 'bert-base-chinese',
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        hidden_size: int = 768,
        unfreeze_layers: int = 0,
        use_pooler: bool = True,
        add_lstm: bool = False,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 2
    ):
        """
        Args:
            model_name: 预训练模型名称或路径
            num_classes: 分类数量
            dropout_rate: Dropout率
            hidden_size: BERT隐藏层大小
            unfreeze_layers: 解冻BERT最后几层（0表示全部冻结）
            use_pooler: 是否使用BERT的pooler输出
            add_lstm: 是否添加LSTM层
            lstm_hidden_size: LSTM隐藏层大小
            lstm_layers: LSTM层数
        """
        super().__init__()
        
        # 加载预训练BERT
        # self.bert = BertModel.from_pretrained(model_name)
        self.bert = AutoModelForMaskedLM.from_pretrained(model_name).bert  
        self.hidden_size = hidden_size
        self.use_pooler = use_pooler
        self.add_lstm = add_lstm
        
        # 冻结/解冻策略
        self._freeze_bert_layers(unfreeze_layers)
        
        # LSTM层（可选）
        if add_lstm:
            self.lstm = nn.LSTM(
                input_size=hidden_size,
                hidden_size=lstm_hidden_size,
                num_layers=lstm_layers,
                bidirectional=True,
                batch_first=True,
                dropout=dropout_rate if lstm_layers > 1 else 0
            )
            classifier_input_size = lstm_hidden_size * 2  # 双向LSTM
        else:
            classifier_input_size = hidden_size
        
        # 分类头
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(classifier_input_size, num_classes)
        
        # 添加额外的全连接层（可选）
        self.pre_classifier = nn.Linear(classifier_input_size, classifier_input_size)
        self.activation = nn.ReLU()
        
        # 初始化权重
        self._init_weights()
    
    def _freeze_bert_layers(self, unfreeze_layers: int):
        """冻结/解冻BERT层"""
        # 首先冻结所有参数
        for param in self.bert.parameters():
            param.requires_grad = False
        
        if unfreeze_layers > 0:
            # 解冻最后n层encoder
            for layer in self.bert.encoder.layer[-unfreeze_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # 解冻pooler层
            if getattr(self.bert, 'pooler', None) is not None:
                for param in self.bert.pooler.parameters():
                    param.requires_grad = True
    
    def _init_weights(self):
        """初始化分类层权重"""
        if hasattr(self, 'pre_classifier'):
            nn.init.xavier_uniform_(self.pre_classifier.weight)
            nn.init.constant_(self.pre_classifier.bias, 0)
        
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
        if self.add_lstm:
            for name, param in self.lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            input_ids: 输入token ids
            attention_mask: 注意力掩码
            token_type_ids: token类型ids
            labels: 标签（训练时使用）
            
        Returns:
            logits: 分类logits
            loss: 损失（如果提供了labels）
        """
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        if self.use_pooler and getattr(outputs, 'pooler_output', None) is not None:
            # 使用[CLS]的pooler输出
            sequence_output = outputs.pooler_output
        else:
            # 使用[CLS]的最后隐藏状态
            sequence_output = outputs.last_hidden_state[:, 0, :]
        
        # LSTM层（如果启用）
        if self.add_lstm:
            lstm_out, _ = self.lstm(outputs.last_hidden_state)
            # 使用最后时刻的输出
            sequence_output = lstm_out[:, -1, :]
            
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # 预分类层
        sequence_output = self.pre_classifier(sequence_output)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.dropout(sequence_output)
        
        # 分类层
        logits = self.classifier(sequence_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return logits, loss
    
    def get_embeddings(self, input_ids, attention_mask, token_type_ids=None):
        """获取文本的嵌入表示（用于相似度计算等）"""
        with torch.no_grad():
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            if self.use_pooler and hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings


class MultiTaskModel(ImprovedModel):
    """多任务学习模型（情感分析 + 情感强度预测）"""
    
    def __init__(
        self,
        model_name: str = 'bert-base-chinese',
        num_sentiment_classes: int = 2,
        num_intensity_classes: int = 3,  # 弱、中、强
        **kwargs
    ):
        super().__init__(model_name=model_name, num_classes=num_sentiment_classes, **kwargs)
        
        # 添加情感强度分类头
        classifier_input_size = self.hidden_size
        self.intensity_classifier = nn.Linear(classifier_input_size, num_intensity_classes)
        
        # 初始化
        nn.init.xavier_uniform_(self.intensity_classifier.weight)
        nn.init.constant_(self.intensity_classifier.bias, 0)
    
    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids=None,
        sentiment_labels=None,
        intensity_labels=None
    ):
        """多任务前向传播"""
        # 获取BERT输出
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        if self.use_pooler and hasattr(outputs, 'pooler_output'):
            sequence_output = outputs.pooler_output
        else:
            sequence_output = outputs.last_hidden_state[:, 0, :]
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # 预分类层
        sequence_output = self.pre_classifier(sequence_output)
        sequence_output = self.activation(sequence_output)
        sequence_output = self.dropout(sequence_output)
        
        # 情感分类
        sentiment_logits = self.classifier(sequence_output)
        
        # 情感强度分类
        intensity_logits = self.intensity_classifier(sequence_output)
        
        # 计算损失
        total_loss = None
        if sentiment_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            sentiment_loss = loss_fct(
                sentiment_logits.view(-1, sentiment_logits.size(-1)),
                sentiment_labels.view(-1)
            )
            total_loss = sentiment_loss
            
            if intensity_labels is not None:
                intensity_loss = loss_fct(
                    intensity_logits.view(-1, intensity_logits.size(-1)),
                    intensity_labels.view(-1)
                )
                total_loss = sentiment_loss + 0.5 * intensity_loss  # 权重可调
        
        return {
            'sentiment_logits': sentiment_logits,
            'intensity_logits': intensity_logits,
            'loss': total_loss
        }


class AttentionModel(ImprovedModel):
    """带注意力机制的模型"""
    
    def __init__(self, model_name='bert-base-chinese', **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=kwargs.get('dropout_rate', 0.3),
            batch_first=True
        )
        
        # 注意力权重的线性层
        self.attention_weights = nn.Linear(self.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """带注意力的前向传播"""
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 获取所有token的表示
        hidden_states = outputs.last_hidden_state
        
        # 应用自注意力
        attended_output, attention_weights = self.attention(
            hidden_states, hidden_states, hidden_states,
            key_padding_mask=~attention_mask.bool()
        )
        
        # 加权平均池化
        weights = self.attention_weights(attended_output)
        weights = torch.softmax(weights, dim=1)
        weighted_output = (attended_output * weights).sum(dim=1)
        
        # 分类
        weighted_output = self.dropout(weighted_output)
        logits = self.classifier(weighted_output)
        
        # 计算损失
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return logits, loss