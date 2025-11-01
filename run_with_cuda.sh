#!/bin/bash
# 自动设置 CUDA 库路径并运行 Python 脚本
# 用法: ./run_with_cuda.sh python src/train.py --dataset book --model_type DNN

echo "================================================"
echo "自动设置 CUDA 库路径"
echo "================================================"
echo ""

# 检查是否在 conda 环境中
if [ -z "$CONDA_PREFIX" ]; then
    echo "❌ 错误：未检测到 conda 环境"
    echo "   请先激活 conda 环境：conda activate comirec"
    exit 1
fi

echo "✅ 检测到 conda 环境: $CONDA_PREFIX"
CONDA_LIB="$CONDA_PREFIX/lib"

# 检查 conda lib 目录是否存在
if [ ! -d "$CONDA_LIB" ]; then
    echo "❌ 错误：conda lib 目录不存在: $CONDA_LIB"
    exit 1
fi

# 检查是否有 CUDA 库
if ! ls "$CONDA_LIB"/libcudart.so* 1> /dev/null 2>&1; then
    echo "⚠️  警告：在 $CONDA_LIB 中未找到 CUDA 库"
    echo "   请确保已安装 CUDA Toolkit："
    echo "   conda install -c conda-forge cudatoolkit=12.4 cudnn"
    exit 1
fi

echo "✅ 找到 CUDA 库目录: $CONDA_LIB"

# 设置 LD_LIBRARY_PATH
CURRENT_LD_PATH="${LD_LIBRARY_PATH:-}"
if [[ "$CURRENT_LD_PATH" != *"$CONDA_LIB"* ]]; then
    if [ -z "$CURRENT_LD_PATH" ]; then
        export LD_LIBRARY_PATH="$CONDA_LIB"
    else
        export LD_LIBRARY_PATH="$CONDA_LIB:$CURRENT_LD_PATH"
    fi
    echo "✅ 已设置 LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
else
    echo "ℹ️  LD_LIBRARY_PATH 已包含 conda lib 路径"
fi

echo ""
echo "================================================"
echo "运行命令: $@"
echo "================================================"
echo ""

# 执行传入的命令
exec "$@"

