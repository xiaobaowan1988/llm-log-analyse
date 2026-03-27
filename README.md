# LLM 日志分析 —— LLaMA-Factory 微调指南

> 使用 LLaMA-Factory 对开源大模型进行 LoRA 微调，打造专属「日志分析专家模型」。

---

## 一、环境要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Linux (Ubuntu 20.04+) | Ubuntu 22.04 |
| Python | 3.9+ | 3.10 |
| GPU | NVIDIA，显存 ≥ 6GB | RTX 3090/4090（24GB）|
| CUDA | 11.8+ | 12.1 |
| 硬盘 | 50GB 可用 | 100GB+ |

> **无 GPU？** 可将 `quantization_bit: 4` 改为 CPU 模式（速度极慢，仅供测试）。

---

## 二、快速开始（5 步跑通 Demo）

### 第 1 步：安装 LLaMA-Factory

```bash
bash scripts/install_llamafactory.sh
```

脚本会自动完成：克隆仓库 → 创建虚拟环境 → 安装所有依赖 → 验证 GPU。

---

### 第 2 步：（可选）国内镜像加速模型下载

首次运行会从 HuggingFace 下载基座模型（约 1GB）。国内网络建议先配置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

### 第 3 步A：启动可视化 WebUI（零代码操作）

```bash
bash scripts/start_webui.sh
```

浏览器打开 **http://localhost:7860**，按如下步骤操作：

```
Train 标签页
  ├── Model Name  → Qwen/Qwen2.5-0.5B-Instruct
  ├── Finetuning  → LoRA
  ├── Dataset     → log_sft_demo
  ├── Template    → qwen
  └── 点击 [Start] 按钮开始训练
```

---

### 第 3 步B：命令行一键训练（适合自动化）

```bash
bash scripts/run_demo_train.sh
```

训练参数说明（可在脚本中调整）：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| 基座模型 | Qwen2.5-0.5B-Instruct | 轻量模型，适合 Demo |
| 微调方法 | LoRA (rank=8) | 显存友好 |
| 量化 | 4-bit QLoRA | 6GB 显存可运行 |
| 训练步数 | 100 steps | 快速验证，~5 分钟 |
| 输出目录 | `output/lora_demo_<时间戳>` | LoRA 权重 |

---

### 第 4 步：推理测试

```bash
bash scripts/run_demo_inference.sh
# 或指定权重目录
bash scripts/run_demo_inference.sh ./output/lora_demo_20240115_120000
```

脚本会自动加载 LoRA 权重并对 3 条典型日志进行分析推理。

---

### 第 5 步：使用配置文件进行完整训练

```bash
source ~/LLaMA-Factory/venv/bin/activate
cd ~/LLaMA-Factory
llamafactory-cli train $(pwd)/../llm-log-analyse/config/lora_sft_config.yaml
```

---

## 三、目录结构

```
llm-log-analyse/
├── scripts/
│   ├── install_llamafactory.sh    # 安装脚本
│   ├── start_webui.sh             # 启动 Gradio WebUI
│   ├── run_demo_train.sh          # 命令行 Demo 训练
│   └── run_demo_inference.sh      # 推理测试
├── dataset/
│   └── log_sft_demo.json          # 示例训练数据（10 条日志问答对）
├── config/
│   └── lora_sft_config.yaml       # 完整训练配置文件
└── output/                        # 训练产出（自动创建）
    └── lora_demo_<timestamp>/
        ├── adapter_config.json    # LoRA 配置
        ├── adapter_model.bin      # LoRA 权重
        └── training_loss.png      # Loss 曲线
```

---

## 四、训练数据格式

数据集使用 Alpaca 格式（JSON 数组），每条样本包含三个字段：

```json
{
  "instruction": "你是一名资深运维专家，请分析以下系统日志...",
  "input": "2024-01-15 03:22:18 ERROR [payment-service] ...",
  "output": "## 根因判断\n...\n## 紧急处理步骤\n...\n## 长期优化建议\n..."
}
```

参考 `dataset/log_sft_demo.json`，按相同格式添加你自己的日志数据即可。

---

## 五、常见问题

**Q：CUDA out of memory？**
- 将 `per_device_train_batch_size` 改为 1
- 将 `lora_rank` 从 16 降为 8
- 确认已开启 `quantization_bit: 4`

**Q：模型下载太慢？**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

**Q：想换更大的模型？**
- 将 `model_name_or_path` 改为 `Qwen/Qwen2.5-7B-Instruct`
- 同步修改 `template: qwen`
- 7B 模型需要 ≥ 10GB 显存（4-bit 量化）

**Q：如何导出模型用于 vLLM / Ollama 部署？**
```bash
cd ~/LLaMA-Factory
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --adapter_name_or_path ./output/lora_demo_xxx \
    --template qwen \
    --finetuning_type lora \
    --export_dir ./exported_model \
    --export_size 4       # 每个分片 4GB
```
