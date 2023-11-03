import os
from pathlib import Path
from typing import Union

from .const import fps_filename, timestamp_filename, imgs_dirname, video_formats
from .dataset import Sequence, ImageSequence, VideoSequence

def is_video_file(filepath: str) -> bool:
    return Path(filepath).suffix.lower() in video_formats

def get_fps_file(dirpath: str) -> Union[None, str]:
    fps_file = os.path.join(dirpath, fps_filename)
    if os.path.isfile(fps_file):
        return fps_file
    return None

def get_timestamp_directory(dirpath: str) -> Union[None, str]:
    """
    用于读取时间戳文件 代替静态fps
    """
    timestamp_dir = os.path.join(dirpath, timestamp_filename)
    if os.path.isfile(timestamp_dir):
        return timestamp_dir
    return None

def get_imgs_directory(dirpath: str) -> Union[None, str]:
    imgs_dir = os.path.join(dirpath, imgs_dirname)
    if os.path.isdir(imgs_dir):
        return imgs_dir
    return None

def get_video_file(dirpath: str) -> Union[None, str]:
    filenames = [f for f in os.listdir(dirpath) if is_video_file(f)]
    if len(filenames) == 0:
        return None
    assert len(filenames) == 1
    filepath = os.path.join(dirpath, filenames[0])
    return filepath

def fps_from_file(fps_file) -> float:
    assert os.path.isfile(fps_file)
    with open(fps_file, 'r') as f:
        fps = float(f.readline().strip())
    assert fps > 0, 'Expected fps to be larger than 0. Instead got fps={}'.format(fps)
    return fps

def get_sequence_or_none(dirpath: str) -> Union[None, Sequence]:
    fps_file = get_fps_file(dirpath)
    if fps_file:
        # Must be a sequence (either ImageSequence or VideoSequence)
        fps = fps_from_file(fps_file)
        imgs_dir = get_imgs_directory(dirpath)
        timestamp_dir = get_timestamp_directory(dirpath)
        if imgs_dir:
            return ImageSequence(imgs_dir, fps, timestamp_dir)
        video_file = get_video_file(dirpath)
        assert video_file is not None
        return VideoSequence(video_file, fps)
    # Can be VideoSequence if there is a video file. But have to use fps from meta data.
    video_file = get_video_file(dirpath)
    if video_file is not None:
        return VideoSequence(video_file)
    return None


