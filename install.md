# 方式1: 直接 pip install（使用更新后的默认架构）
pip install -e .

# 方式2: 指定 CUDA 架构编译
python setup.py build_ext --cuda-archs="70;75" bdist_wheel
pip install dist/*.whl

# 方式3: 仅 V100
python setup.py build_ext --cuda-archs="70" bdist_wheel

# 方式4: 仅 T600/Turing  
python setup.py build_ext --cuda-archs="75" bdist_wheel