import os
import shutil
import pickle as pkl
import numpy as np
import cv2
from tqdm import tqdm


def read_event_timestamp(i, events_path):
    event = np.load((events_path + '/' + str(i).zfill(10) + '.npz'), allow_pickle=True)
    event_t = event[event.files[2]][0]
    return event_t


if __name__ == '__main__':

    # Dataset_dir = "I:/Dataset/Avgkitti/data_odometry_gray/dataset"
    Dataset_dir = "/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset"
    events_dir = Dataset_dir + "/events"
    # events_dir = "F:/Dataset/Avgkitti/data_odometry_gray/dataset" + "/events"
    img_dir = Dataset_dir + "/images"
    timestamp_dir = Dataset_dir + "/imageFiles_Upsample"

    # Data/Events
    # cnt = 0
    for event_list in os.listdir(events_dir):
        # if cnt == 0:
        if 1:
            event_path = events_dir + '/' + event_list
            print('Generating events timestamp of ' + str(event_list))

            img_len = len(os.listdir((events_dir + '/' + event_list)))
            pbar = tqdm(total=img_len)

            timestamp_list = []
            for i in range(img_len):        # Timestamps for each label in the sequence
                event_t = read_event_timestamp(i, event_path)
                timestamp_list.append("{:.10f}".format(event_t*1e-9))   # ns
                pbar.update(1)
            if not os.path.exists(timestamp_dir + '/' + event_list):
                os.makedirs(timestamp_dir + '/' + event_list)
            with open((timestamp_dir + '/' + event_list + '/timestamps.txt'), 'w') as file:
                for timestamp in timestamp_list:
                    file.write(f"{timestamp}\n")

            pbar.close()
        # cnt += 1
