import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# 缩放点积注意力
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                q: torch.Tensor,  # 查询 q[batch_size, n_heads, seq_len,d_k]
                k: torch.Tensor,  # 键 k[batch_size, n_heads, seq_len,d_k]
                v: torch.Tensor,  # 值 v[batch_size, n_heads, seq_len,d_k]
                mask: torch.Tensor = None  # 掩码 mask[batch_size,1,seq_len,seq_len]
                ) -> tuple[torch.Tensor, torch.Tensor]:
        # 计算Q.K^T的点积
        d_k = q.size(-1)
        # scores 单个token对每个token的打分
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # [batch_size, n_heads, seq_len, seq_len]

        # 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)  # 掩码位置设为极小值，softmax后接近0

        # 计算注意力权重(softmax) 并dropout
        attn_weights = self.dropout(F.softmax(scores, dim=-1))

        # 加权求和得到输出
        output = torch.matmul(attn_weights, v)

        return output, attn_weights


# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被n_heads整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 定义Q/K/V的线性投影层(输入输出维度都是d_model)
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # 输出投影层
        self.w_o = nn.Linear(d_model, d_model)

        # 缩放点积注意力
        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                q: torch.Tensor,  # 查询 q[batch_size, seq_len,d_model]
                k: torch.Tensor,  # 键 k[batch_size, seq_len,d_model]
                v: torch.Tensor,  # 值 v[batch_size, seq_len,d_model]
                mask: torch.Tensor = None  # 注意力掩码
                ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = q.size(0)

        # 1. 线性投影拆分多头
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, n_heads, d_k) -> (batch_size, n_heads, seq_len, d_k)
        q = self.w_q(q).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 2. 计算缩放点积注意力
        attn_output, attn_weights = self.attention(q, k, v, mask)

        # 3. 拼接多头结果
        # (batch_size, n_heads, seq_len, d_k) -> (batch_size, seq_len, n_heads * d_k) = (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 4. 输出投影 + dropout
        return self.dropout(self.w_o(attn_output)), attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # 第一层升维
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x