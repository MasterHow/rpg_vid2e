import os
import shutil
import pickle as pkl
import numpy as np
import cv2
from shutil import copy
import shutil
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset_dir",
                        default="/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset")
    parser.add_argument("--output_prefix", default='images_reorganize_cuda0')
    # parser.add_argument("--seq", default=['04'])
    parser.add_argument('--seq', nargs='+', type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = get_args()

    Dataset_dir = args.Dataset_dir
    input_dir = Dataset_dir + "/sequences"
    output_dir = Dataset_dir + '/' + args.output_prefix
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # files = os.listdir(input_dir)
    # files.sort(key=lambda x: int(x.split('.')[0]))
    # cnt = 0       # 第几个序列（00，01，02……）
    # files = ['08', '09', '10']      # 指定序列
    # files = ['04']  # 指定序列
    files = args.seq  # 指定序列
    for Seq_list in files:
        Seq_list = str(Seq_list).zfill(2)
        # if cnt == 0:      # 指定调整的序列
        #     cnt += 1
        # elif cnt <= 11:
        # for i in range(2):      # 左右眼
        #     if i == 0:
        #         cam_view = 'image_0'
        #     elif i == 1:
        #         cam_view = 'image_1'

        cam_view = 'image_0'    # 暂时只生成左目数据
        Seqinpath = input_dir + '/' + Seq_list + '/' + cam_view + '/'
        Seqoutpath = output_dir + '/' + Seq_list + '_' + cam_view + '/imgs/'

        if not os.path.exists(Seqoutpath):
            os.makedirs(Seqoutpath)

        print('Reorganizing... : ' + Seqinpath)
        pbar = tqdm(total=len(os.listdir(Seqinpath)))
        for img_list in os.listdir(Seqinpath):
            img = cv2.imread(Seqinpath + img_list)
            cv2.imwrite((Seqoutpath + "/" + img_list), img)
            pbar.update(1)
        pbar.close()

        # copy_dir(Seqinpath, Seqoutpath)
        file = open(output_dir + '/' + Seq_list + '_' + cam_view + '/fps.txt', 'w')
        file.write("9.64")
        file.close()

        # 复制时间戳文件
        source_file = input_dir + '/' + Seq_list + '/' + 'times.txt'  # 源文件路径
        destination_file = output_dir + '/' + Seq_list + '_' + cam_view + '/' + 'times.txt'  # 目标文件路径
        shutil.copy2(source_file, destination_file)

        # cnt += 1
