# Proj 因子生成Agent

本项目用于构建一个面向量化因子研究的大模型 Agent，整体分为三个阶段：

- `extracter`：从研报中抽取高质量单因子样本
- `SFT`：用结构化样本进行监督微调，注入因子研究范式
- `DPO`：后续进行偏好对齐，使输出更贴近真实研究偏好

## 目录说明

- `extracter/`：研报解析、候选发现、样本生成与校验
- `SFT/`：SFT 数据准备、训练配置与训练脚本
- `研报/`：原始研报 PDF 数据
- `SPEC.md`：项目总设计
- `requirements.txt`：Ubuntu + NVIDIA GPU 环境依赖

## 快速开始

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 运行 Extracter

```bash
python3 -m extracter.cli --stage discovery
python3 -m extracter.cli --stage generate
```

3. 运行 SFT 数据准备

```bash
python3 -m SFT.cli --stage m1
python3 -m SFT.cli --stage m2
```

4. 云端启动 SFT 训练

```bash
python3 -m SFT.cli --stage m3 --train-config SFT/configs/train_config.yaml
```

## 文档入口

- 项目总设计：[`SPEC.md`](/Users/king/Documents/硕士/Sem2/8307Natural_language_processing_and_text_analytics/Proj_v2/SPEC.md)
- Extracter 设计文档：[`extracter/README.md`](/Users/king/Documents/硕士/Sem2/8307Natural_language_processing_and_text_analytics/Proj_v2/extracter/README.md)
- SFT 设计文档：[`SFT/README.md`](/Users/king/Documents/硕士/Sem2/8307Natural_language_processing_and_text_analytics/Proj_v2/SFT/README.md)

## 当前状态

- Extracter 已完成基础数据抽取链路
- SFT 已完成 M1、M2，并已切到 `TRL` 训练框架
- DPO 仍待实现
