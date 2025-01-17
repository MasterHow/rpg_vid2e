# -*- coding: utf-8 -*-
# @Time    : 2022/10/10 9:48
# @Author  : Jiaan Chen, Hao Shi
# @Purpose : Generate event gray data.
import glob
import os
import numpy as np
from tqdm import tqdm
import cv2
import argparse
# from my_extension.sigma3norm.normalize_image import normalizeImage3Sigma_cython


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset_dir",
                        default="/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset")
    parser.add_argument('--seq', nargs='+', type=int)
    args = parser.parse_args()
    return args


def multidim_evframe_gen(data, imageH=260, imageW=346):
    # x = data[0, :]  # x
    # y = data[1, :]  # y
    # t = data[2, :]  # t
    # p = data[3, :]  # p [0, 1]

    x = event[:, 0]
    y = event[:, 1]
    t = event[:, 2]
    p = event[:, 3]
    p = np.where(p == -1, 0, p)     # 将负事件的极性转换为0，与原处理方式保持一致

    num_events = len(x)
    if num_events > 0:
        # t_ref = t[-1]  # time of the last event in the packet
        # tau = 50000  # decay parameter (in micro seconds)
        # tau = (t[-1] - t[0]) / 2      # cja使用的是这个
        img_size = (imageH, imageW)
        img_pos = np.zeros(img_size, np.int)
        img_neg = np.zeros(img_size, np.int)
        # sae_pos = np.zeros(img_size, np.float32)
        # sae_neg = np.zeros(img_size, np.float32)
        # cnt = np.zeros(img_size, np.float32)
        # sae = np.zeros(img_size, np.float32)
        # 创建用于记录最后一个正事件和负事件时间戳的数组
        last_positive_event_timestamp = np.zeros(img_size, dtype=float)
        last_negative_event_timestamp = np.zeros(img_size, dtype=float)
        tau = t[-1] - t[0]      # 我们与EV-FlowNet保持一致将最后一个时刻的事件时间戳归一化到最大为1

        # # 原版代码 循环操作效率低
        # for idx in range(num_events):
        #     coordx = int(x[idx])
        #     coordy = int(y[idx])
        #     if p[idx] > 0:
        #         img_pos[coordy, coordx] += 1  # count events
        #         # sae_pos[coordy, coordx] = np.exp(-(t_ref - t[idx]) / tau)
        #     else:
        #         img_neg[coordy, coordx] += 1
        #         # sae_neg[coordy, coordx] = np.exp(-(t_ref - t[idx]) / tau)
        #     # cnt[coordy, coordx] += 1
        #     # sae[coordy, coordx] = np.exp(-(t_ref - t[idx]) / tau)

        # # 用于检验优化前后代码的结果是否一致
        # img_pos_temp = img_pos
        # img_neg_temp = img_neg
        # img_pos = np.zeros(img_size, np.int)
        # img_neg = np.zeros(img_size, np.int)

        # 优化代码速度 计算指数值，然后将其放入适当的位置
        # exp_vals = np.exp(-(t_ref - t) / tau)
        # img_pos[y, x] += (p > 0) * exp_vals
        # img_neg[y, x] += (p <= 0) * exp_vals
        # cnt[y, x] += 1
        # 根据事件的极性累积事件计数
        positive_events = p > 0
        negative_events = p <= 0
        np.add.at(img_pos, (y[positive_events], x[positive_events]), 1)
        np.add.at(img_neg, (y[negative_events], x[negative_events]), 1)
        # 记录最后一个事件的时间戳
        last_positive_event_timestamp[y[positive_events], x[positive_events]] = t[positive_events] / tau
        last_negative_event_timestamp[y[negative_events], x[negative_events]] = t[negative_events] / tau

        # cnt_sae = np.multiply(cnt, sae)
        # img_pos = normalizeImage3Sigma(img_pos, imageH=imageH, imageW=imageW)
        # img_neg = normalizeImage3Sigma(img_neg, imageH=imageH, imageW=imageW)
        # # sae_pos = normalizeImage3Sigma(sae_pos, imageH=imageH, imageW=imageW)
        # # sae_neg = normalizeImage3Sigma(sae_neg, imageH=imageH, imageW=imageW)
        # # cnt_sae = normalizeImage3Sigma(cnt_sae, imageH=imageH, imageW=imageW)
        # cnt = normalizeImage3Sigma(cnt, imageH=imageH, imageW=imageW)
        # # sae = normalizeImage3Sigma(sae, imageH=imageH, imageW=imageW)

        img_pos = normalizeImage3Sigma_v2(img_pos, imageH=imageH, imageW=imageW)
        img_neg = normalizeImage3Sigma_v2(img_neg, imageH=imageH, imageW=imageW)
        last_positive_event_timestamp = normalizeImage3Sigma_v2(last_positive_event_timestamp,
                                                                imageH=imageH, imageW=imageW)
        last_negative_event_timestamp = normalizeImage3Sigma_v2(last_negative_event_timestamp,
                                                                imageH=imageH, imageW=imageW)
        # cnt = normalizeImage3Sigma_v2(cnt, imageH=imageH, imageW=imageW)
        # img_pos = normalizeImage3Sigma_v3(img_pos, imageH=imageH, imageW=imageW)
        # img_neg = normalizeImage3Sigma_v3(img_neg, imageH=imageH, imageW=imageW)
        # cnt = normalizeImage3Sigma_v3(cnt, imageH=imageH, imageW=imageW)

        # md_evframe = np.concatenate((img_pos[:, :, np.newaxis], img_neg[:, :, np.newaxis],
        #                              sae_pos[:, :, np.newaxis], sae_neg[:, :, np.newaxis],
        #                              cnt_sae[:, :, np.newaxis], cnt[:, :, np.newaxis], sae[:, :, np.newaxis]), axis=2)
        # 只保存正负事件的积累数量,以及最后一个时刻的事件时间戳
        md_evframe = np.concatenate((img_pos[:, :, np.newaxis],
                                     img_neg[:, :, np.newaxis],
                                     last_positive_event_timestamp[:, :, np.newaxis],
                                     last_negative_event_timestamp[:, :, np.newaxis]), axis=2)

        md_evframe = md_evframe.astype(np.uint8)
    else:
        img_size = (imageH, imageW)
        img_pos = np.zeros(img_size, np.int)
        img_neg = np.zeros(img_size, np.int)
        # sae_pos = np.zeros(img_size, np.float32)
        # sae_neg = np.zeros(img_size, np.float32)
        # cnt = np.zeros(img_size, np.float32)
        # sae = np.zeros(img_size, np.float32)
        # cnt_sae = np.multiply(cnt, sae)
        last_positive_event_timestamp = np.zeros(img_size, dtype=float)
        last_negative_event_timestamp = np.zeros(img_size, dtype=float)

        # md_evframe = np.concatenate((img_pos[:, :, np.newaxis], img_neg[:, :, np.newaxis],
        #                              sae_pos[:, :, np.newaxis], sae_neg[:, :, np.newaxis],
        #                              cnt_sae[:, :, np.newaxis], cnt[:, :, np.newaxis], sae[:, :, np.newaxis]), axis=2)
        # 只保存正负事件的积累数量,以及最后一个时刻的事件时间戳
        md_evframe = np.concatenate((img_pos[:, :, np.newaxis],
                                     img_neg[:, :, np.newaxis],
                                     last_positive_event_timestamp[:, :, np.newaxis],
                                     last_negative_event_timestamp[:, :, np.newaxis]), axis=2)
        md_evframe = md_evframe.astype(np.uint8)

    return md_evframe


def normalizeImage3Sigma(image, imageH=260, imageW=346):
    """
    followed by matlab dhp19 generate
    根据均值和方差对图像进行归一化，同时去除了统计中>3 sigma的离群异常点
    除以3 sigma以更强烈地剪切或者限制图像中的亮度范围，以去除更多的异常值
    没有使用均值图像 和原始DHP19的处理方法一致
    """
    sum_img = np.sum(image)
    count_image = np.sum(image > 0)
    mean_image = sum_img / count_image  # 均值图像
    var_img = np.var(image[image > 0])
    sig_img = np.sqrt(var_img)  # 标准差图像

    if sig_img < 0.1 / 255:
        sig_img = 0.1 / 255

    numSDevs = 3.0
    meanGrey = 0
    range_old = numSDevs * sig_img      # 3 sigma标准
    half_range = 0
    range_new = 255

    normalizedMat = np.zeros([imageH, imageW])
    for i in range(imageH):
        for j in range(imageW):
            l = image[i, j]
            if l == 0:
                normalizedMat[i, j] = meanGrey
            else:
                f = (l + half_range) * range_new / range_old
                if f > range_new:
                    f = range_new

                if f < 0:
                    f = 0
                normalizedMat[i, j] = np.floor(f)

    return normalizedMat


def normalizeImage3Sigma_v2(image, imageH=260, imageW=346):
    sum_img = np.sum(image)
    count_image = np.sum(image > 0)
    mean_image = sum_img / count_image
    var_img = np.var(image[image > 0])
    sig_img = np.sqrt(var_img)

    if sig_img < 0.1 / 255:
        sig_img = 0.1 / 255

    numSDevs = 3.0
    meanGrey = 0
    range_old = numSDevs * sig_img
    half_range = 0
    range_new = 255

    # 使用矩阵操作替代循环
    normalizedMat = np.where(image == 0, meanGrey, (image + half_range) * range_new / range_old)
    normalizedMat = np.clip(normalizedMat, 0, range_new)
    normalizedMat = np.floor(normalizedMat).astype(np.uint8)

    return normalizedMat


# def normalizeImage3Sigma_v3(image, imageH=260, imageW=346):
#     image = np.asarray(image, dtype=np.float64)
#     normalizedMat = normalizeImage3Sigma_cython(image)
#     return normalizedMat


if __name__ == '__main__':
    args = get_args()

    # Dataset_dir = "/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset"
    Dataset_dir = args.Dataset_dir
    events_dir = Dataset_dir + "/events_final"
    img_dir = Dataset_dir + "/sequences"
    timestamp_dir = Dataset_dir + "/imageFiles_Upsample"
    output_dir = Dataset_dir + "/event_cnt_frames"

    # Data/Events
    files = args.seq  # 指定序列
    img_folder_prefix = 'image_0'   # 用于指定左右目文件夹
    for Seq_list in files:
        Seq_list = str(Seq_list).zfill(2)
        event_path = events_dir + '/' + Seq_list + '/' + img_folder_prefix
        output_path = output_dir + '/' + Seq_list + '/' + img_folder_prefix
        print('Generating event  frames of ' + str(Seq_list) + '/' + str(img_folder_prefix))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # 读取图像帧数量
        f = open((img_dir + '/' + Seq_list + '/times.txt'), "r")
        text = f.readlines()
        img_stamp_file = np.array([line.strip("\n") for line in text], dtype=np.float)
        f.close()
        img_len = len(img_stamp_file)

        pbar = tqdm(total=img_len)
        for i in range(img_len):  # for each label in the sequence
            event = np.load((event_path + '/' + str(i).zfill(6) + '.npy'), allow_pickle=True)
            # data = multidim_evframe_gen(event_t, imageH=376, imageW=1241)
            # data = multidim_evframe_gen(event, imageH=376, imageW=1241)
            data = multidim_evframe_gen(event, imageH=352, imageW=1216)     # 此前resize到32的倍数
            np.save(output_path + '/' + str(i).zfill(6) + '.npy', data)
            # cv2.imwrite(output_path + '/' + str(i).zfill(6) + '.jpg', data)
            pbar.update(1)

        pass

    # # 左右目同时生成 Data/Events
    # img_folder_prefix = 'image_0'   # 用于指定左右目文件夹
    # for event_list in sorted(os.listdir(events_dir)):
    #     event_path = events_dir + '/' + event_list + '/' + img_folder_prefix
    #     output_path = output_dir + '/' + event_list + '/' + img_folder_prefix
    #     print('Generating event  frames of ' + str(event_list) + str(img_folder_prefix))
    #
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #
    #     # 读取图像帧数量
    #     f = open((img_dir + '/' + event_list.split('_')[0] + '/times.txt'), "r")
    #     text = f.readlines()
    #     img_stamp_file = np.array([line.strip("\n") for line in text], dtype=np.float)
    #     f.close()
    #     img_len = len(img_stamp_file)
    #
    #     pbar = tqdm(total=img_len)
    #     for i in range(img_len):  # for each label in the sequence
    #         event = np.load((event_path + '/' + str(i).zfill(6) + '.npy'), allow_pickle=True)
    #         # data = multidim_evframe_gen(event_t, imageH=376, imageW=1241)
    #         # data = multidim_evframe_gen(event, imageH=376, imageW=1241)
    #         data = multidim_evframe_gen(event, imageH=352, imageW=1216)     # 此前resize到32的倍数
    #
    #         # 平移data来对齐事件和图像
    #         pass
    #
    #         # np.save(output_path + '/' + str(i).zfill(6) + '.npy', data)
    #         cv2.imwrite(output_path + '/' + str(i).zfill(6) + '.jpg', data)
    #         pbar.update(1)
    #
    #     pass

