#!/usr/bin/env bash
# =============================================================================
# LLaMA-Factory 推理测试脚本
# 加载训练好的 LoRA 权重，对日志进行分析推理
#
# 使用方式：
#   bash scripts/run_demo_inference.sh [LoRA 权重目录]
#   bash scripts/run_demo_inference.sh ./output/lora_demo_20240101_120000
# =============================================================================
set -e

INSTALL_DIR="${HOME}/LLaMA-Factory"
VENV_DIR="${INSTALL_DIR}/venv"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LORA_DIR="${1:-}"

if [ -z "$LORA_DIR" ]; then
    # 自动使用最新的训练输出
    LORA_DIR=$(find "${PROJECT_DIR}/output" -maxdepth 1 -type d -name "lora_demo_*" \
               2>/dev/null | sort | tail -1)
fi

if [ -z "$LORA_DIR" ] || [ ! -d "$LORA_DIR" ]; then
    echo "[ERROR] 未找到 LoRA 权重目录。"
    echo "        请先运行 bash scripts/run_demo_train.sh"
    echo "        或手动指定路径: bash scripts/run_demo_inference.sh <path>"
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
cd "$INSTALL_DIR"

echo ""
echo "========================================"
echo "  加载模型并进行日志分析推理"
echo "  基座模型 : Qwen/Qwen2.5-0.5B-Instruct"
echo "  LoRA 权重: $LORA_DIR"
echo "========================================"
echo ""

# 国内环境加速下载（如有需要，去掉注释）
# export HF_ENDPOINT=https://hf-mirror.com

python3 - <<PYEOF
import sys, os
sys.path.insert(0, "src")

from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

# 加载模型参数
args = {
    "model_name_or_path":  "Qwen/Qwen2.5-0.5B-Instruct",
    "adapter_name_or_path": "${LORA_DIR}",
    "template":            "qwen",
    "finetuning_type":     "lora",
    "quantization_bit":    4,
    "infer_dtype":         "auto",
}

print("[INFO] 正在加载模型，请稍候...")
model = ChatModel(args)
print("[INFO] 模型加载完成！\n")

# 测试用例：模拟几条典型业务日志
test_cases = [
    {
        "instruction": "你是一名运维专家，请分析以下日志并给出根因判断和处理建议。",
        "input": (
            "2024-01-15 03:22:18 ERROR [payment-service] "
            "java.lang.OutOfMemoryError: Java heap space\n"
            "    at java.util.Arrays.copyOf(Arrays.java:3210)\n"
            "    at com.example.PaymentProcessor.processBatch(PaymentProcessor.java:158)\n"
            "堆内存使用率: 98.7% | GC 耗时: 45s | 活跃线程数: 312"
        ),
    },
    {
        "instruction": "你是一名运维专家，请分析以下日志并给出根因判断和处理建议。",
        "input": (
            "2024-01-15 14:05:33 ERROR [api-gateway] "
            "Connect timeout: upstream order-service (10.0.1.23:8080) "
            "after 5000ms\n"
            "失败率: 100% | 持续时长: 8min | 影响接口: /api/v1/orders"
        ),
    },
    {
        "instruction": "你是一名运维专家，请分析以下日志并给出根因判断和处理建议。",
        "input": (
            "2024-01-15 22:48:01 WARN [db-pool] "
            "Connection pool exhausted. Waiting for available connection...\n"
            "pool.size=20, active=20, idle=0, waiting=47\n"
            "慢查询数/分钟: 156 | 最长等待: 12.3s"
        ),
    },
]

print("=" * 60)
for i, case in enumerate(test_cases, 1):
    print(f"\n【测试用例 {i}】")
    print(f"日志输入:\n{case['input']}\n")
    print("模型分析:")

    messages = [{"role": "user", "content": f"{case['instruction']}\n\n{case['input']}"}]
    response = ""
    for token in model.stream_chat(messages):
        response += token
        print(token, end="", flush=True)

    print("\n" + "-" * 60)
    torch_gc()

print("\n[INFO] 推理测试完成！")
PYEOF
