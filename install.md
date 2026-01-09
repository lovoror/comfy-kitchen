# 方式1: 直接 pip install（使用更新后的默认架构）
pip install -e .

# 方式2: 使用环境变量指定 CUDA 架构（推荐）
# 注意：bdist_wheel 命令行参数可能不传递，所以使用环境变量更可靠
COMFY_CUDA_ARCHS="70" pip install -e .

# 方式3: 仅 V100 (SM 7.0) - 完整编译命令
rm -rf build/ dist/ *.egg-info
COMFY_CUDA_ARCHS="70" python setup.py bdist_wheel
pip install dist/*.whl

# 方式4: V100 + T600/Turing
COMFY_CUDA_ARCHS="70;75" pip install -e .
  
# 方式5: 仅安装 CPU 版本（跳过 CUDA 编译，使用 Triton 后端）
pip install -e . --no-cuda