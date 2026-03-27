#!/usr/bin/env bash
# =============================================================================
# LLaMA-Factory 命令行 Demo 训练脚本
# 任务：使用 LoRA 对 Qwen2.5-0.5B-Instruct 进行 SFT 微调
# 数据：日志分析示例数据集（dataset/log_sft_demo.json）
# 硬件要求：单张 GPU，显存 >= 6GB（可按注释调整至 4GB）
#
# 使用方式：bash scripts/run_demo_train.sh
# =============================================================================
set -e

INSTALL_DIR="${HOME}/LLaMA-Factory"
VENV_DIR="${INSTALL_DIR}/venv"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --------------------------------------------------------------------------
# 检查安装
# --------------------------------------------------------------------------
if [ ! -d "$INSTALL_DIR" ]; then
    echo "[ERROR] 未找到 LLaMA-Factory，请先运行 bash scripts/install_llamafactory.sh"
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
cd "$INSTALL_DIR"

# --------------------------------------------------------------------------
# 数据集注册：将示例数据集告知 LLaMA-Factory
# --------------------------------------------------------------------------
DEMO_DATA_SRC="${PROJECT_DIR}/dataset/log_sft_demo.json"
DEMO_DATA_DST="${INSTALL_DIR}/data/log_sft_demo.json"
DATASET_INFO="${INSTALL_DIR}/data/dataset_info.json"

if [ ! -f "$DEMO_DATA_DST" ]; then
    echo "[INFO] 复制示例数据集到 LLaMA-Factory data 目录..."
    cp "$DEMO_DATA_SRC" "$DEMO_DATA_DST"
fi

# 向 dataset_info.json 注册数据集（若尚未注册）
python3 - <<'EOF'
import json, os

info_path = os.path.expanduser("~/LLaMA-Factory/data/dataset_info.json")

with open(info_path) as f:
    info = json.load(f)

if "log_sft_demo" not in info:
    info["log_sft_demo"] = {
        "file_name": "log_sft_demo.json",
        "columns": {
            "prompt":    "instruction",
            "query":     "input",
            "response":  "output"
        }
    }
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print("[INFO] 已注册数据集: log_sft_demo")
else:
    print("[INFO] 数据集 log_sft_demo 已注册，跳过。")
EOF

# --------------------------------------------------------------------------
# 训练参数说明（可按需调整）
# --------------------------------------------------------------------------
# model_name_or_path : 基座模型，首次运行会自动从 HuggingFace 下载
#                      国内环境可换为镜像站：
#                      export HF_ENDPOINT=https://hf-mirror.com
# finetuning_type    : lora（显存友好）；full = 全量微调（需更多显存）
# lora_rank          : LoRA 秩，越大效果越好但显存消耗越多，推荐 8~64
# quantization_bit   : 4 = QLoRA（4-bit 量化，显存可低至 4GB）
#                      注释掉此行则使用 bf16 全精度 LoRA
# per_device_train_batch_size : 单卡 Batch Size，显存不足时调小
# gradient_accumulation_steps: 梯度累积步数，等效全局 batch = batch * 此值
# max_steps          : 训练总步数（Demo 用 100 步快速验证）
# output_dir         : 训练产出（LoRA 权重 + 训练日志）保存路径
# --------------------------------------------------------------------------

OUTPUT_DIR="${PROJECT_DIR}/output/lora_demo_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "========================================"
echo "  开始 Demo 训练"
echo "  模型  : Qwen/Qwen2.5-0.5B-Instruct"
echo "  方法  : LoRA (rank=8)"
echo "  数据  : log_sft_demo (50 条样本)"
echo "  步数  : 100 steps"
echo "  输出  : $OUTPUT_DIR"
echo "========================================"
echo ""

# 国内环境加速下载（如有需要，去掉注释）
# export HF_ENDPOINT=https://hf-mirror.com

python src/train.py \
    --stage                         sft                             \
    --do_train                      True                            \
    --model_name_or_path            Qwen/Qwen2.5-0.5B-Instruct     \
    --dataset                       log_sft_demo                    \
    --template                      qwen                            \
    --finetuning_type               lora                            \
    --lora_rank                     8                               \
    --lora_target                   all                             \
    --quantization_bit              4                               \
    --cutoff_len                    1024                            \
    --per_device_train_batch_size   2                               \
    --gradient_accumulation_steps   4                               \
    --lr_scheduler_type             cosine                          \
    --learning_rate                 1e-4                            \
    --max_steps                     100                             \
    --warmup_steps                  10                              \
    --logging_steps                 10                              \
    --save_steps                    50                              \
    --fp16                          True                            \
    --output_dir                    "$OUTPUT_DIR"                   \
    --report_to                     none

echo ""
echo "========================================"
echo "  训练完成！LoRA 权重保存在："
echo "  $OUTPUT_DIR"
echo ""
echo "  下一步 - 运行推理测试："
echo "    bash scripts/run_demo_inference.sh $OUTPUT_DIR"
echo "========================================"
