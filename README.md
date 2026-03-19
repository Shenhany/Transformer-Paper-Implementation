# Transformer 从零实现：对照《Attention Is All You Need》论文的入门教程

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

本项目面向**深度学习/NLP 初学者**，完全对照 Transformer 经典论文《Attention Is All You Need》，**分步拆解并实现 Transformer 的每一个核心模块**，代码逐行添加详细注释，精准对应论文中的公式、术语和逻辑，帮助新手从「理解论文原理」到「落地可运行代码」。

## 📚 项目定位
- ✨ 零基础友好：无需复杂前置知识，注释覆盖所有关键概念，代码模块化拆分，便于逐块学习；
- 📝 论文对齐：每个模块的实现逻辑严格对应《Attention Is All You Need》原文；
- 🧩 模块化设计：核心组件拆分到独立文件，可单独查看/调试，降低学习门槛；
- 🚀 开箱即用：所有代码可直接运行，附带测试用例验证模块功能。

## 📋 实现内容（按论文逻辑排序）
| 模块名称                | 对应文件               | 核心说明                                  |
|-------------------------|------------------------|-------------------------------------------|
| 词嵌入层（TokenEmbedding） | `embeddings.py`        | 实现词嵌入 + 根号d_model缩放（论文3.4节） |
| 位置编码层（PositionalEncoding） | `embeddings.py` | 实现正弦/余弦位置编码（论文3.5节）|
| 缩放点积注意力（ScaledDotProductAttention） | `attention.py` | 实现注意力核心计算 + 掩码机制（论文3.2.1节） |
| 多头注意力（MultiHeadAttention） | `attention.py` | 实现多头拆分/拼接 + 线性投影（论文3.2.2节） |
| 位置前馈网络（PositionWiseFeedForward） | `attention.py` | 实现升维/降维 + ReLU激活（论文3.3节）|
| 编码器层/解码器层       | `transformer.py`       | 实现残差连接 + 层归一化（论文3.1节）|
| 完整 Transformer 模型   | `transformer.py`       | 编码器+解码器+输出投影（论文3.4节）|

## 🛠 环境准备
### 1. 依赖安装
```bash
# 核心依赖（PyTorch 需根据系统/显卡适配，建议2.0+）
pip install torch>=2.0.0 numpy>=1.24.0
