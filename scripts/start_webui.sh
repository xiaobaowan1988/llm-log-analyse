#!/usr/bin/env bash
# =============================================================================
# 启动 LLaMA-Factory WebUI（Gradio 可视化训练界面）
# 使用方式：bash scripts/start_webui.sh
# =============================================================================
set -e

INSTALL_DIR="${HOME}/LLaMA-Factory"
VENV_DIR="${INSTALL_DIR}/venv"

# --------------------------------------------------------------------------
# 检查安装目录
# --------------------------------------------------------------------------
if [ ! -d "$INSTALL_DIR" ]; then
    echo "[ERROR] 未找到 LLaMA-Factory，请先运行 bash scripts/install_llamafactory.sh"
    exit 1
fi

# --------------------------------------------------------------------------
# 激活虚拟环境
# --------------------------------------------------------------------------
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

cd "$INSTALL_DIR"

# --------------------------------------------------------------------------
# 将项目的数据集软链接到 LLaMA-Factory 的 data 目录
# --------------------------------------------------------------------------
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_SRC="${PROJECT_DIR}/dataset"
DATASET_DST="${INSTALL_DIR}/data/log_analyse"

if [ -d "$DATASET_SRC" ]; then
    mkdir -p "$(dirname "$DATASET_DST")"
    if [ ! -L "$DATASET_DST" ]; then
        ln -s "$DATASET_SRC" "$DATASET_DST"
        echo "[INFO] 已链接数据集目录: $DATASET_DST -> $DATASET_SRC"
    fi
fi

# --------------------------------------------------------------------------
# 启动 WebUI
# --------------------------------------------------------------------------
# 参数说明：
#   GRADIO_SHARE=0        - 不生成公网分享链接（内网使用改为 1）
#   --host 0.0.0.0        - 允许局域网访问（仅本机访问改为 127.0.0.1）
#   --port 7860           - 默认端口
# --------------------------------------------------------------------------
echo ""
echo "========================================"
echo "  启动 LLaMA-Factory WebUI"
echo "  访问地址: http://localhost:7860"
echo "  按 Ctrl+C 停止服务"
echo "========================================"
echo ""

GRADIO_SHARE=0 python src/webui.py \
    --host 0.0.0.0 \
    --port 7860
