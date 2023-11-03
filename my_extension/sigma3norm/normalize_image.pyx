# normalize_image.pyx
import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t  # 添加这行导入

def normalizeImage3Sigma_cython(np.ndarray[np.float64_t, ndim=2] image):
    cdef int imageH = image.shape[0]
    cdef int imageW = image.shape[1]
    cdef double sum_img = np.sum(image)
    cdef double count_image = np.sum(image > 0)
    cdef double mean_image = sum_img / count_image
    cdef double var_img = np.var(image[image > 0])
    cdef double sig_img = np.sqrt(var_img)

    if sig_img < 0.1 / 255:
        sig_img = 0.1 / 255

    cdef double numSDevs = 3.0
    cdef double meanGrey = 0.0
    cdef double range_old = numSDevs * sig_img
    cdef double half_range = 0.0
    cdef double range_new = 255.0

    cdef np.ndarray[np.uint8_t, ndim=2] normalizedMat = np.zeros((imageH, imageW), dtype=np.uint8)

    cdef double[:, :] image_view = image
    cdef uint8_t[:, :] normalizedMat_view = normalizedMat

    for i in range(imageH):
        for j in range(imageW):
            l = image_view[i, j]
            if l == 0:
                normalizedMat_view[i, j] = <uint8_t>f
            else:
                f = (l + half_range) * range_new / range_old
                if f > range_new:
                    f = range_new
                if f < 0:
                    f = 0
                normalizedMat_view[i, j] = <uint8_t>f

    return normalizedMat
