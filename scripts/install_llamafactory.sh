#!/usr/bin/env bash
# =============================================================================
# LLaMA-Factory 安装脚本
# 适用环境：Linux + NVIDIA GPU (CUDA 11.8 / 12.1)
# 使用方式：bash scripts/install_llamafactory.sh
# =============================================================================
set -e

echo "========================================"
echo "  LLaMA-Factory 环境安装脚本"
echo "========================================"

# --------------------------------------------------------------------------
# 0. 基础依赖检查
# --------------------------------------------------------------------------
check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "[ERROR] 未找到命令: $1，请先安装后重试。"
        exit 1
    fi
}

check_command python3
check_command git
check_command pip3

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[INFO] 当前 Python 版本: $PYTHON_VERSION"

# 要求 Python >= 3.9
python3 -c "
import sys
if sys.version_info < (3, 9):
    print('[ERROR] 需要 Python >= 3.9，当前版本不满足要求。')
    sys.exit(1)
"

# --------------------------------------------------------------------------
# 1. 克隆 LLaMA-Factory 仓库
# --------------------------------------------------------------------------
INSTALL_DIR="${HOME}/LLaMA-Factory"

if [ -d "$INSTALL_DIR" ]; then
    echo "[INFO] 目录 $INSTALL_DIR 已存在，跳过克隆，执行 git pull 更新..."
    git -C "$INSTALL_DIR" pull
else
    echo "[INFO] 正在克隆 LLaMA-Factory..."
    git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"

# --------------------------------------------------------------------------
# 2. 创建并激活虚拟环境（推荐隔离安装）
# --------------------------------------------------------------------------
VENV_DIR="${INSTALL_DIR}/venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] 创建 Python 虚拟环境..."
    python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
echo "[INFO] 已激活虚拟环境: $VENV_DIR"

# --------------------------------------------------------------------------
# 3. 升级 pip 并安装 LLaMA-Factory 及依赖
# --------------------------------------------------------------------------
echo "[INFO] 升级 pip..."
pip install --upgrade pip -q

echo "[INFO] 安装 LLaMA-Factory（含所有可选依赖）..."
# extras:
#   metrics  - 评估指标 (rouge, nltk)
#   deepspeed - 多卡 / ZeRO 分布式训练
#   bitsandbytes - 4/8-bit 量化 (QLoRA)
#   vllm     - 高速推理后端
pip install -e ".[metrics,deepspeed,bitsandbytes,vllm]" -q

# --------------------------------------------------------------------------
# 4. 验证安装
# --------------------------------------------------------------------------
echo ""
echo "[INFO] 验证核心依赖..."
python3 - <<'EOF'
import importlib, sys
packages = {
    "torch":            "PyTorch",
    "transformers":     "Transformers",
    "peft":             "PEFT (LoRA)",
    "trl":              "TRL (RLHF/DPO)",
    "datasets":         "Datasets",
    "gradio":           "Gradio (WebUI)",
    "bitsandbytes":     "BitsAndBytes (量化)",
}
all_ok = True
for pkg, name in packages.items():
    try:
        mod = importlib.import_module(pkg)
        ver = getattr(mod, "__version__", "unknown")
        print(f"  [OK] {name:<25} {ver}")
    except ImportError:
        print(f"  [MISSING] {name}")
        all_ok = False

import torch
if torch.cuda.is_available():
    print(f"\n  [GPU] 检测到 {torch.cuda.device_count()} 块 GPU")
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem  = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"       GPU {i}: {name}  ({mem:.1f} GB)")
else:
    print("\n  [WARN] 未检测到可用 GPU，将以 CPU 模式运行（速度极慢）。")

sys.exit(0 if all_ok else 1)
EOF

echo ""
echo "========================================"
echo "  安装完成！"
echo ""
echo "  激活环境命令："
echo "    source ${VENV_DIR}/bin/activate"
echo ""
echo "  下一步："
echo "    bash scripts/start_webui.sh      # 启动可视化界面"
echo "    bash scripts/run_demo_train.sh   # 运行命令行 Demo 训练"
echo "========================================"
