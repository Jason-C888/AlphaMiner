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

4. 下载基础模型到本地

```bash
python3 -m SFT.cli --stage download --train-config SFT/configs/train_config.yaml
```

说明：

- 下载目标模型由 [`SFT/configs/train_config.yaml`](/Users/king/Documents/硕士/Sem2/8307Natural_language_processing_and_text_analytics/Proj_v2/SFT/configs/train_config.yaml) 中的 `model.model_id` 指定
- 本地保存路径由同一配置中的 `model.local_model_dir` 指定
- 如果跳过这一步，训练阶段会在发现本地模型不存在时自动先下载

5. 启动 SFT 训练

```bash
python3 -m SFT.cli --stage m3 --train-config SFT/configs/train_config.yaml
```

6. 运行推理或评估

```bash
python3 -m SFT.cli --stage infer --inference-config SFT/configs/inference_config.yaml --inspiration "这里填写研究现象描述"
python3 -m SFT.cli --stage eval --inference-config SFT/configs/inference_config.yaml
```

## 关键调用

- `python3 -m SFT.cli --stage download --train-config SFT/configs/train_config.yaml`
  - 按训练配置下载基础模型到本地目录
- `python3 -m SFT.cli --stage m3 --train-config SFT/configs/train_config.yaml`
  - 启动训练；优先从本地基础模型目录加载，缺失时自动下载
- `python3 -m SFT.download_model --config SFT/configs/train_config.yaml`
  - 独立下载脚本，功能与 `--stage download` 等价
- `python3 -m SFT.cli --stage infer --inference-config SFT/configs/inference_config.yaml --inspiration "..."`
  - 按推理配置生成单条因子结果
- `python3 -m SFT.cli --stage eval --inference-config SFT/configs/inference_config.yaml`
  - 用推理配置批量评估 `SFT/data/test.jsonl` 或配置中指定的数据集

## 目录约定

- `model/base/`：基础预训练模型本地目录
- `trained/`：训练输出目录，保存 LoRA adapter、tokenizer 和运行 manifest
- `SFT/data/`：SFT 训练、验证、测试数据
- `SFT/output/`：数据准备、评估、人工复核等输出

默认约定：

- 基础模型默认下载到 `model/base/Qwen3-0.6B`
- 训练产物默认保存到 `trained/qwen3_0_6b_lora`

## 推理配置提示

[`SFT/configs/inference_config.yaml`](/Users/king/Documents/硕士/Sem2/8307Natural_language_processing_and_text_analytics/Proj_v2/SFT/configs/inference_config.yaml) 支持两种后端：

- `openai_compat`：走远程 API
- `local_hf`：走本地 Hugging Face 模型目录，可配合 `base_model_path + adapter_path` 加载本地底座模型和训练产物

## 文档入口

- 项目总设计：[`SPEC.md`](/Users/king/Documents/硕士/Sem2/8307Natural_language_processing_and_text_analytics/Proj_v2/SPEC.md)
- Extracter 设计文档：[`extracter/README.md`](/Users/king/Documents/硕士/Sem2/8307Natural_language_processing_and_text_analytics/Proj_v2/extracter/README.md)
- SFT 设计文档：[`SFT/README.md`](/Users/king/Documents/硕士/Sem2/8307Natural_language_processing_and_text_analytics/Proj_v2/SFT/README.md)

## 当前状态

- Extracter 已完成基础数据抽取链路
- SFT 已完成 M1、M2，并已切到 `TRL` 训练框架
- DPO 仍待实现
