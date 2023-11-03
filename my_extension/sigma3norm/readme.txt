Install:
pip install cython
apt-get install python3-dev
python setup.py build_ext --inplace

Use:
import numpy as np
from my_extension.sigma3norm.normalize_image import normalizeImage3Sigma_cython

def normalizeImage3Sigma(image, imageH=260, imageW=346):
    image = np.asarray(image, dtype=np.float64)
    normalizedMat = normalizeImage3Sigma_cython(image)
    return normalizedMat

# 示例用法
image = np.random.rand(260, 346) * 255  # 随机生成一个260x346的图像
normalized_image = normalizeImage3Sigma(image)

貌似速度没有np快。。。