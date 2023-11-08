import os
import shutil
import pickle as pkl
import numpy as np
import cv2
import argparse
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--Dataset_dir",
                        default="/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset")
    parser.add_argument('--seq', nargs='+', type=int)
    args = parser.parse_args()
    return args


def read_event_timestamp(i, events_path):
    event = np.load((events_path + '/' + str(i).zfill(10) + '.npz'), allow_pickle=True)
    event_t = event[event.files[2]][0]
    return event_t


if __name__ == '__main__':
    args = get_args()

    # Dataset_dir = "/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset"
    Dataset_dir = args.Dataset_dir
    events_dir = Dataset_dir + "/events"
    timestamp_dir = Dataset_dir + "/imageFiles_Upsample"

    # Data/Events
    files = args.seq  # 指定序列
    for Seq_list in files:
        Seq_list = str(Seq_list).zfill(2)
        event_path = events_dir + '/' + Seq_list + '_image_0'
        print('Generating events timestamp of ' + str(Seq_list + '_image_0'))

        img_len = len(os.listdir((events_dir + '/' + Seq_list + '_image_0')))
        pbar = tqdm(total=img_len)

        timestamp_list = []
        for i in range(img_len):  # Timestamps for each label in the sequence
            event_t = read_event_timestamp(i, event_path)
            timestamp_list.append("{:.10f}".format(event_t * 1e-9))  # ns
            pbar.update(1)
        if not os.path.exists(timestamp_dir + '/' + Seq_list + '/' + 'image_0'):
            os.makedirs(timestamp_dir + '/' + Seq_list + '/' + 'image_0')
        with open((timestamp_dir + '/' + Seq_list + '/' + 'image_0' + '/timestamps.txt'), 'w') as file:
            for timestamp in timestamp_list:
                file.write(f"{timestamp}\n")

        pbar.close()

    # 左右目同时生成
    # for event_list in os.listdir(events_dir):
    #     event_path = events_dir + '/' + event_list
    #     print('Generating events timestamp of ' + str(event_list))
    #
    #     img_len = len(os.listdir((events_dir + '/' + event_list)))
    #     pbar = tqdm(total=img_len)
    #
    #     timestamp_list = []
    #     for i in range(img_len):        # Timestamps for each label in the sequence
    #         event_t = read_event_timestamp(i, event_path)
    #         timestamp_list.append("{:.10f}".format(event_t*1e-9))   # ns
    #         pbar.update(1)
    #     if not os.path.exists(timestamp_dir + '/' + event_list):
    #         os.makedirs(timestamp_dir + '/' + event_list)
    #     with open((timestamp_dir + '/' + event_list + '/timestamps.txt'), 'w') as file:
    #         for timestamp in timestamp_list:
    #             file.write(f"{timestamp}\n")
    #
    #     pbar.close()
