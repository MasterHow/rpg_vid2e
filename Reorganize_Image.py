import os
import shutil
import pickle as pkl
import numpy as np
import cv2
from shutil import copy

if __name__ == '__main__':

    Dataset_dir = "/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset"
    input_dir = Dataset_dir + "/sequences"
    # output_dir = Dataset_dir + "/images"
    output_dir = Dataset_dir + "/images_reorganize_cuda1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # files = os.listdir(input_dir)
    # files.sort(key=lambda x: int(x.split('.')[0]))
    # cnt = 0       # 第几个序列（00，01，02……）
    # files = ['08', '09', '10']      # 指定序列
    files = ['00']  # 指定序列
    for Seq_list in files:
        # if cnt == 0:      # 指定调整的序列
        #     cnt += 1
        # elif cnt <= 11:
        for i in range(2):      # 左右眼
            if i == 0:
                cam_view = 'image_0'
            elif i == 1:
                cam_view = 'image_1'
            Seqinpath = input_dir + '/' + Seq_list + '/' + cam_view + '/'
            Seqoutpath = output_dir + '/' + Seq_list + '_' + cam_view + '/imgs/'

            if not os.path.exists(Seqoutpath):
                os.makedirs(Seqoutpath)
            for img_list in os.listdir(Seqinpath):
                img = cv2.imread(Seqinpath + img_list)
                cv2.imwrite((Seqoutpath + "/" + img_list), img)

            # copy_dir(Seqinpath, Seqoutpath)
            file = open(output_dir + '/' + Seq_list + '_' + cam_view + '/fps.txt', 'w')
            file.write("9.64")
            file.close()

            # cnt += 1
