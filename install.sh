#!/usr/bin/env bash
# DynaMask-VIO installer — works with conda and micromamba
# Usage: bash install.sh [cuda|rocm|cpu]
set -euo pipefail

BACKEND="${1:-auto}"
CONDA_CMD=""

# Detect conda or micromamba
if command -v micromamba &>/dev/null; then
    CONDA_CMD="micromamba"
elif command -v mamba &>/dev/null; then
    CONDA_CMD="mamba"
elif command -v conda &>/dev/null; then
    CONDA_CMD="conda"
else
    echo "ERROR: No conda/micromamba/mamba found. Install one first."
    exit 1
fi
echo "Using: $CONDA_CMD"

# Auto-detect GPU backend
if [ "$BACKEND" = "auto" ]; then
    if command -v nvcc &>/dev/null || [ -d /usr/local/cuda ]; then
        BACKEND="cuda"
        echo "Auto-detected: CUDA"
    elif command -v rocminfo &>/dev/null || [ -d /opt/rocm ]; then
        BACKEND="rocm"
        echo "Auto-detected: ROCm"
    else
        BACKEND="cpu"
        echo "Auto-detected: CPU (no GPU toolkit found)"
    fi
fi

# Create conda environment
echo "Creating environment..."
$CONDA_CMD env create -f environment.yml -y 2>/dev/null || \
    $CONDA_CMD env update -f environment.yml

# Activate (for script context, we use run)
ACTIVATE="$CONDA_CMD run -n dynamask"

# Install PyTorch for the correct backend
echo "Installing PyTorch for backend: $BACKEND"
case "$BACKEND" in
    cuda)
        # Detect CUDA version
        if command -v nvcc &>/dev/null; then
            CUDA_VER=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
            CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
            CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        else
            CUDA_MAJOR=12
            CUDA_MINOR=1
        fi

        if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
            TORCH_URL="https://download.pytorch.org/whl/cu124"
        else
            TORCH_URL="https://download.pytorch.org/whl/cu121"
        fi
        $ACTIVATE pip install torch">=2.0.0" torchvision">=0.15.0" --index-url "$TORCH_URL"
        $ACTIVATE pip install pypose">=0.6.0"
        ;;
    rocm)
        $ACTIVATE pip install torch">=2.0.0" torchvision">=0.15.0" --index-url https://download.pytorch.org/whl/rocm6.2
        $ACTIVATE pip install pypose">=0.6.0"
        ;;
    cpu)
        $ACTIVATE pip install torch">=2.0.0" torchvision">=0.15.0" --index-url https://download.pytorch.org/whl/cpu
        $ACTIVATE pip install pypose">=0.6.0"
        ;;
    *)
        echo "Unknown backend: $BACKEND (use cuda, rocm, or cpu)"
        exit 1
        ;;
esac

echo ""
echo "Done! Activate with:  $CONDA_CMD activate dynamask"
