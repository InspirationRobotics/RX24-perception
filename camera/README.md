# Useful links

https://docs.ultralytics.com/guides/nvidia-jetson/

https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
Note: For torchvision just install with pip3 install torchvision.whl

Add this to bashrc:
export LD_PRELOAD=/lib/aarch64-linux-gnu/libstdc++.so.6:$LD_PRELOAD