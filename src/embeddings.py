import torch
import torch.nn as nn
import numpy as np
import math


# 词嵌入层
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor):
        return self.embedding(x) * math.sqrt(self.d_model)  # 符合原文乘以根号d_model


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()

        # 根据传入 dropout 参数定义dropout层
        self.dropout = nn.Dropout(p=dropout)

        # 初始化位置编码矩阵
        pe = torch.zeros(max_len, d_model)  # 生成全零初始
        # 生成位置索引：(max_len,1) -> 每个位置对应一个索引(0,1...,max_len - 1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # 计算分母项
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        # 填充位置编码: 偶数位用sin, 奇数位用cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 增加batch 维度: (1,max_len,d_model) -> 方便广播到整个batch
        pe = pe.unsqueeze(0)  # 保持batch_first

        # 注册为buffer: 不会参与计算(位置编码是固定规则，无需优化)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 只取和输入序列长度匹配的位置编码(避免过长的pe)
        x = x + self.pe[:, :x.size(1), :]
        # 加dropout,增加鲁棒性
        return self.dropout(x)