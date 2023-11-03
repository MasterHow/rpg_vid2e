import matplotlib.pyplot as plt
import numpy as np
import os

def viz_events(events, resolution):
    pos_events = events[events[:,-1]==1]
    neg_events = events[events[:,-1]==-1]

    image_pos = np.zeros(resolution[0]*resolution[1], dtype="uint8")
    image_neg = np.zeros(resolution[0]*resolution[1], dtype="uint8")

    np.add.at(image_pos, (pos_events[:,0]+pos_events[:,1]*resolution[1]).astype("int32"), pos_events[:,-1]**2)
    np.add.at(image_neg, (neg_events[:,0]+neg_events[:,1]*resolution[1]).astype("int32"), neg_events[:,-1]**2)

    # image_rgb = np.stack(
    #     [
    #         image_pos.reshape(resolution),
    #         image_neg.reshape(resolution),
    #         np.zeros(resolution, dtype="uint8")
    #     ], -1
    # ) * 50
    image_rgb = np.stack(
        [
            image_pos.reshape(resolution),
            np.zeros(resolution, dtype="uint8"),
            image_neg.reshape(resolution),
        ], -1
    ) * 50

    return image_rgb


if __name__ == '__main__':

    # Dataset_dir = "I:/Dataset/Avgkitti/data_odometry_gray/dataset"
    Dataset_dir = "/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset"
    events_dir = Dataset_dir + "/events_final"
    output_dir = Dataset_dir + "/events_show"

    cnt = 0
    resolution = [352, 1216]
    for event_list in os.listdir(events_dir):
        event_path = events_dir + '/' + event_list
        for cam_view in os.listdir(event_path):
            if cnt == 0:
                cam_path = event_path + '/' + cam_view
                output_path = output_dir + '/' + event_list + '/' + cam_view

                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                for name in os.listdir(cam_path):
                    # events = np.load('I:/Dataset/Avgkitti/data_odometry_gray/dataset/events_final/00_image_0_000001.npy')
                    events = np.load((cam_path + '/' + name))
                    image_rgb = viz_events(events, resolution)
                    handle = plt.imshow(image_rgb)
                    plt.axis('off')
                    plt.savefig((output_path + '/' + name.split('.')[0] + '.png'), dpi=1000, bbox_inches='tight')
                    # plt.show(block=False)
                    plt.close()

            cnt += 1
