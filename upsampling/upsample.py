import argparse
import os
# Must be set before importing torch.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

from utils import Upsampler


def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help='Path to input directory. See README.md for expected structure of the directory.')
    parser.add_argument("--output_dir", required=True, help='Path to non-existing output directory. This script will generate the directory.')
    args = parser.parse_args()
    return args


def main():
    flags = get_flags()

    input_dir = flags.input_dir
    output_dir = flags.output_dir
    count = 0
    for Seq_list in sorted(os.listdir(input_dir)):
        Seqinpath = input_dir + '/' + Seq_list + '/'
        Seqoutpath = output_dir + '/' + Seq_list + '/'
        upsampler = Upsampler(input_dir=Seqinpath, output_dir=Seqoutpath)
        upsampler.upsample()


if __name__ == '__main__':
    main()
