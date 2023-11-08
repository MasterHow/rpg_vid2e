import os
import shutil
import pickle as pkl
import numpy as np
import cv2
from tqdm import tqdm
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset_dir",
                        default="/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset")
    parser.add_argument('--seq', nargs='+', type=int)
    args = parser.parse_args()
    return args


def merge_event(event_stamp, img_stamp, i, events_path):    # Merge event according to the current timestamp
    start_stamp = (img_stamp[i] + img_stamp[i + 1]) / 2
    end_stamp = (img_stamp[i + 1] + img_stamp[i + 2]) / 2
    indexes = np.array(np.where((event_stamp > start_stamp) * (event_stamp <= end_stamp))).reshape(-1)
    event = np.load((events_path + '/' + str(indexes[0]).zfill(10) + '.npz'), allow_pickle=True)
    event_x = event[event.files[0]]
    event_y = event[event.files[1]]
    event_t = event[event.files[2]]
    event_p = event[event.files[3]]
    count = 0
    for index in indexes:
        if count > 0:
            try:
                event_index = np.load((events_path + '/' + str(index).zfill(10) + '.npz'), allow_pickle=True)
                event_x = np.append(event_x, event_index[event_index.files[0]])
                event_y = np.append(event_y, event_index[event_index.files[1]])
                event_t = np.append(event_t, event_index[event_index.files[2]])
                event_p = np.append(event_p, event_index[event_index.files[3]])
            except:
                print(('Fail in '+ events_path + '/' + str(index).zfill(10) + '.npz'))
                pass
        count += 1
    return event_x, event_y, event_t, event_p


def merge_event_first(event_stamp, img_stamp, events_path):    # Merge event according to the current timestamp
    start_stamp = img_stamp[0]
    end_stamp = (img_stamp[0] + img_stamp[1]) / 2
    indexes = np.array(np.where((event_stamp > start_stamp) * (event_stamp <= end_stamp))).reshape(-1)
    event = np.load((events_path + '/' + str(indexes[0]).zfill(10) + '.npz'), allow_pickle=True)
    event_x = event[event.files[0]]
    event_y = event[event.files[1]]
    event_t = event[event.files[2]]
    event_p = event[event.files[3]]
    count = 0
    for index in indexes:
        if count > 0:
            try:
                event_index = np.load((events_path + '/' + str(index).zfill(10) + '.npz'), allow_pickle=True)
                event_x = np.append(event_x, event_index[event_index.files[0]])
                event_y = np.append(event_y, event_index[event_index.files[1]])
                event_t = np.append(event_t, event_index[event_index.files[2]])
                event_p = np.append(event_p, event_index[event_index.files[3]])
            except:
                print(('Fail in ' + events_path + '/' + str(index).zfill(10) + '.npz'))
                pass
        count += 1
    return event_x, event_y, event_t, event_p


def merge_event_last(event_stamp, img_stamp, events_path):    # Merge event according to the current timestamp
    start_stamp = (img_stamp[-2] + img_stamp[-1]) / 2
    end_stamp = img_stamp[-1]
    indexes = np.array(np.where((event_stamp > start_stamp) * (event_stamp <= end_stamp))).reshape(-1)
    event = np.load((events_path + '/' + str(indexes[0]).zfill(10) + '.npz'), allow_pickle=True)
    event_x = event[event.files[0]]
    event_y = event[event.files[1]]
    event_t = event[event.files[2]]
    event_p = event[event.files[3]]
    count = 0
    for index in indexes:
        if count > 0:
            try:
                event_index = np.load((events_path + '/' + str(index).zfill(10) + '.npz'), allow_pickle=True)
                event_x = np.append(event_x, event_index[event_index.files[0]])
                event_y = np.append(event_y, event_index[event_index.files[1]])
                event_t = np.append(event_t, event_index[event_index.files[2]])
                event_p = np.append(event_p, event_index[event_index.files[3]])
            except:
                print(('Fail in ' + events_path + '/' + str(index).zfill(10) + '.npz'))
                pass
        count += 1
    return event_x, event_y, event_t, event_p


if __name__ == '__main__':
    args = get_args()

    # Dataset_dir = "/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset"
    Dataset_dir = args.Dataset_dir
    events_dir = Dataset_dir + "/events"
    # events_dir = "F:/Dataset/Avgkitti/data_odometry_gray/dataset" + "/events"
    # img_dir = Dataset_dir + "/images"
    img_dir = Dataset_dir + "/sequences"
    timestamp_dir = Dataset_dir + "/imageFiles_Upsample"
    output_dir = Dataset_dir + "/events_final"

    # Data/Events
    # fps = 9.64
    # files = os.listdir(events_dir)  # 所有序列
    files = args.seq  # 指定序列
    for Seq_list in files:
        Seq_list = str(Seq_list).zfill(2)
        event_list = events_dir + '/' + Seq_list + '_image_0'

        # event_path = events_dir + '/' + event_list
        event_path = events_dir + '/' + Seq_list + '_image_0'
        # output_path = output_dir + '/' + event_list.split('_')[0] + '/' + event_list.split('_')[1] + '_' + event_list.split('_')[2]
        output_path = output_dir + '/' + Seq_list + '/image_0'
        print('Generating events of ' + str(event_list))

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # img_len = len(os.listdir((img_dir + '/' + event_list + '/imgs')))
        # f = open((img_dir + '/' + event_list.split('_')[0] + '/times.txt'), "r")
        f = open((img_dir + '/' + Seq_list + '/times.txt'), "r")
        text = f.readlines()
        img_stamp_file = np.array([line.strip("\n") for line in text], dtype=np.float)
        f.close()
        img_len = len(img_stamp_file)
        # img_stamp = np.arange(0, img_len * (1 / fps), 1 / fps)
        img_stamp = img_stamp_file

        # f = open((timestamp_dir + '/' + event_list + '/timestamps.txt'), "r")
        f = open((timestamp_dir + '/' + Seq_list + '/image_0' + '/timestamps.txt'), "r")
        text = f.readlines()
        event_stamp = np.array([line.strip("\n") for line in text], dtype=np.float)[0:-1]
        f.close()

        pbar = tqdm(total=img_len)
        # 生成第一帧和最后一帧
        event_x, event_y, event_t, event_p = merge_event_first(event_stamp, img_stamp, event_path)
        event = np.concatenate((event_x.reshape(-1, 1), event_y.reshape(-1, 1), event_t.reshape(-1, 1),
                                event_p.reshape(-1, 1)), axis=1)
        np.save((output_path + '/' + str(0).zfill(6) + '.npy'), event)
        pbar.update(1)

        event_x, event_y, event_t, event_p = merge_event_last(event_stamp, img_stamp, event_path)
        event = np.concatenate((event_x.reshape(-1, 1), event_y.reshape(-1, 1), event_t.reshape(-1, 1),
                                event_p.reshape(-1, 1)), axis=1)
        np.save((output_path + '/' + str(img_len-1).zfill(6) + '.npy'), event)
        pbar.update(1)

        for i in range(img_len - 2):        # Timestamps for each label in the sequence
            event_x, event_y, event_t, event_p = merge_event(event_stamp, img_stamp, i, event_path)
            event = np.concatenate((event_x.reshape(-1, 1), event_y.reshape(-1, 1), event_t.reshape(-1, 1),
                                    event_p.reshape(-1, 1)), axis=1)
            np.save((output_path + '/' + str(i+1).zfill(6)+'.npy'), event)

            pbar.update(1)

        pbar.close()


    # 所有序列
    # for event_list in os.listdir(events_dir):
    #     event_path = events_dir + '/' + event_list
    #     output_path = output_dir + '/' + event_list.split('_')[0] + '/' + event_list.split('_')[1] + '_' + event_list.split('_')[2]
    #     print('Generating events of ' + str(event_list))
    #
    #     if not os.path.exists(output_path):
    #         os.makedirs(output_path)
    #
    #     # img_len = len(os.listdir((img_dir + '/' + event_list + '/imgs')))
    #     f = open((img_dir + '/' + event_list.split('_')[0] + '/times.txt'), "r")
    #     text = f.readlines()
    #     img_stamp_file = np.array([line.strip("\n") for line in text], dtype=np.float)
    #     f.close()
    #     img_len = len(img_stamp_file)
    #     # img_stamp = np.arange(0, img_len * (1 / fps), 1 / fps)
    #     img_stamp = img_stamp_file
    #
    #     f = open((timestamp_dir + '/' + event_list + '/timestamps.txt'), "r")
    #     text = f.readlines()
    #     event_stamp = np.array([line.strip("\n") for line in text], dtype=np.float)[0:-1]
    #     f.close()
    #
    #     pbar = tqdm(total=img_len)
    #     # 生成第一帧和最后一帧
    #     event_x, event_y, event_t, event_p = merge_event_first(event_stamp, img_stamp, event_path)
    #     event = np.concatenate((event_x.reshape(-1, 1), event_y.reshape(-1, 1), event_t.reshape(-1, 1),
    #                             event_p.reshape(-1, 1)), axis=1)
    #     np.save((output_path + '/' + str(0).zfill(6) + '.npy'), event)
    #     pbar.update(1)
    #
    #     event_x, event_y, event_t, event_p = merge_event_last(event_stamp, img_stamp, event_path)
    #     event = np.concatenate((event_x.reshape(-1, 1), event_y.reshape(-1, 1), event_t.reshape(-1, 1),
    #                             event_p.reshape(-1, 1)), axis=1)
    #     np.save((output_path + '/' + str(img_len-1).zfill(6) + '.npy'), event)
    #     pbar.update(1)
    #
    #     for i in range(img_len - 2):        # Timestamps for each label in the sequence
    #         event_x, event_y, event_t, event_p = merge_event(event_stamp, img_stamp, i, event_path)
    #         event = np.concatenate((event_x.reshape(-1, 1), event_y.reshape(-1, 1), event_t.reshape(-1, 1),
    #                                 event_p.reshape(-1, 1)), axis=1)
    #         np.save((output_path + '/' + str(i+1).zfill(6)+'.npy'), event)
    #
    #         pbar.update(1)
    #
    #     pbar.close()
