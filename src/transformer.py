import torch
import torch.nn as nn
from attention import MultiHeadAttention, PositionWiseFeedForward
from embeddings import TokenEmbedding, PositionalEncoding


# 残差连接
class ResNet(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        # Pre-LN 结构：先归一化 -> 子层计算 -> dropout -> 残差连接
        return x + self.dropout(sublayer(self.norm(x)))


# 编码层
class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 d_ff: int,
                 dropout: float = 0.1):
        super().__init__()
        # 多头注意力
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        # 残差连接
        self.resnet1 = ResNet(d_model, dropout)
        self.resnet2 = ResNet(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        # 多头注意力机制
        x = self.resnet1(x, lambda x: self.self_attn(x, x, x, src_mask)[0])
        # 前馈网络
        x = self.resnet2(x, self.feed_forward)
        return x


# 编码器(多层编码层堆叠而来)
class Encoder(nn.Module):
    def __init__(self,
                 vocab_size: int,  # 词表大小
                 d_model: int,  # 模型维度
                 n_heads: int,  # 多头注意力头数
                 d_ff: int,  # 前馈网络中间维度
                 n_layers: int,  # 编码层数
                 max_len: int,  # 最大长度
                 dropout: float = 0.1):
        super().__init__()
        # 词嵌入层
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        # 位置编码层
        self.position_embedding = PositionalEncoding(d_model, max_len, dropout)
        # 堆叠N个编码层
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # 层归一化层
        self.norm = nn.LayerNorm(d_model)

    def forward(self,
                x: torch.Tensor,
                src_mask: torch.Tensor = None
                ) -> torch.Tensor:
        # 词嵌入 + 位置编码
        x = self.token_embedding(x)
        x = self.position_embedding(x)

        # 依次通过所有编码器层
        for layer in self.layers:
            x = layer(x, src_mask)

        # 归一化返回
        return self.norm(x)

# 解码层
class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model:int,
                 n_heads:int,
                 d_ff:int,
                 dropout:float = 0.1):
        super().__init__()
        # 掩码多头注意力机制
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 交叉注意力机制
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        # 前馈网络
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        # 残差连接 这里定义了三个
        self.resnet1 = ResNet(d_model, dropout)
        self.resnet2 = ResNet(d_model, dropout)
        self.resnet3 = ResNet(d_model, dropout)

    def forward(self,
                x:torch.Tensor,         # 解码器输入
                enc_output:torch.Tensor,    # 编码器输出
                tgt_mask:torch.Tensor,      # 解码器自注意力掩码(遮挡未来)
                src_tgt_mask: torch.Tensor  # 交叉注意力掩码(遮挡编码器padding)
                ) -> torch.Tensor:
        # 掩码自注意力 (Q=K=V=解码器输入x)
        x = self.resnet1(x, lambda x: self.self_attn(x, x, x, tgt_mask)[0])
        # 交叉注意力 (Q=解码器输入x, K=V=编码器输出enc_output) 即模型推理过程看原文
        x = self.resnet2(x, lambda x: self.cross_attn(x, enc_output, enc_output, src_tgt_mask)[0])
        # 前馈网络
        x = self.resnet3(x, self.feed_forward)
        return x

# 完整解码器 (多层解码层打包)
class Decoder(nn.Module):
    def __init__(self,
                 vocab_size:int,
                 d_model:int = 512,     # 模型维度
                 n_heads:int = 8,       # 多头注意力头数
                 d_ff:int = 2048,       # 前馈网络中间维度
                 n_layers:int = 6,      # 解码层数
                 max_len:int = 5000,    # 最大长度
                 dropout:float = 0.1    # 随机失活概率
                 ):
        super().__init__()
        # 词嵌入层  (和编码器共享独立访问均可，这里独立)
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        # 位置编码
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        # 堆叠N个解码层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        # 归一化
        self.norm = nn.LayerNorm(d_model)

    def forward(self,
                x:torch.Tensor,             # 解码输入
                enc_output:torch.Tensor,    # 编码器输出
                tgt_mask:torch.Tensor,      # 解码器自注意力掩码
                src_tgt_mask:torch.Tensor   # 交叉注意力掩码
                ) -> torch.Tensor:
        # 词嵌入 + 位置编码
        x = self.token_embedding(x)
        x = self.pos_encoding(x)

        # 循环调用解码层
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, src_tgt_mask)

        # 归一化返回
        return self.norm(x)

# 打包后的完整transformer
class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size: int,   # 源语言词表大小
                 tgt_vocab_size: int,   # 目标语言词表大小
                 d_model: int = 512,    # 模型维度
                 n_heads: int = 8,      # 多头注意力头数
                 d_ff: int = 2048,      # 前馈网络中间维度
                 n_layer: int = 6,      # 编码/解码层数
                 max_len: int = 5000,    # 输入序列最大长度
                 dropout: float = 0.1   # 随机失活概率
                 ):
        super().__init__()

        # 编码器
        self.encoder = Encoder(src_vocab_size, d_model, n_heads, d_ff, n_layer, max_len, dropout)
        # 解码器
        self.decoder = Decoder(tgt_vocab_size, d_model, n_heads, d_ff, n_layer, max_len, dropout)

        # 输出映射层
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self,
                src: torch.Tensor,      # 源语言输入序列
                tgt: torch.Tensor,      # 目标语言输入序列
                src_mask:torch.Tensor = None,   # 源语言自注意力掩码
                tgt_mask:torch.Tensor = None,   # 目标语言自注意力掩码
                src_tgt_mask:torch.Tensor = None # 交叉注意力掩码
                ) -> torch.Tensor:
        # 编码器前向传播
        enc_output = self.encoder(src, src_mask)
        # 解码器前向传播
        dec_output = self.decoder(tgt, enc_output, tgt_mask, src_tgt_mask)
        # 输出映射层
        output = self.fc_out(dec_output)
        return output


# 测试模型
if __name__ == "__main__":
    # 超参数
    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    n_heads = 8
    d_ff = 2048
    n_layers = 2

    # 初始化模型
    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_heads, d_ff, n_layers)

    # 构造输入（batch_size=2, src_seq_len=10, tgt_seq_len=8）
    src = torch.randint(0, src_vocab_size, (2, 10))
    tgt = torch.randint(0, tgt_vocab_size, (2, 8))

    # 前向传播
    output = model(src, tgt)
    print(f"输出形状: {output.shape}")  # 预期: (2, 8, 1000)