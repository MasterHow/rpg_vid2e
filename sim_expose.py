import cv2
import os
import glob
import random
import argparse
from tqdm import tqdm


def overexpose_image(img_path):
    """
    对图片进行过度曝光处理，并使效果在一定范围内随机浮动。
    """
    img = cv2.imread(img_path)
    # 随机生成亮度和对比度调整值
    alpha = random.uniform(1.5, 2.0)  # 亮度控制在 1.5 到 2.0 之间
    beta = random.randint(50, 100)    # 对比度控制在 50 到 100 之间
    overexposed_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return overexposed_img


def underexpose_image(img_path):
    """
    对图片进行欠曝光处理，并使效果在一定范围内随机浮动。
    """
    img = cv2.imread(img_path)
    # 随机生成亮度和对比度调整值
    alpha = random.uniform(0.5, 0.8)  # 亮度控制在 0.5 到 0.8 之间
    beta = random.randint(-100, -50)  # 对比度控制在 -100 到 -50 之间
    underexposed_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return underexposed_img


def process_images(input_folder, output_folder, exposure_type='over'):
    """
    处理指定文件夹下的所有 PNG 图片，并保存到输出文件夹。
    根据 exposure_type 选择进行过度曝光或欠曝光处理。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 使用 glob.glob 获取图片路径列表
    img_paths = glob.glob(input_folder + '/**/*.png', recursive=True)

    for img_path in tqdm(img_paths, desc="处理图片: " + input_folder):
        if exposure_type == 'over':
            processed_img = overexpose_image(img_path)
        elif exposure_type == 'under':
            processed_img = underexpose_image(img_path)
        else:
            raise ValueError("Invalid exposure type. Choose 'over' or 'under'.")

        # 创建输出路径并保存图片
        relative_path = os.path.relpath(img_path, input_folder)
        output_path = os.path.join(output_folder, relative_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, processed_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量处理图片并应用曝光效果")
    parser.add_argument("--input_folder", required=True, help="输入图片文件夹的路径")
    parser.add_argument("--output_folder", required=True, help="输出图片文件夹的路径")
    parser.add_argument("--exposure_type", choices=['over', 'under'], required=True,
                        help="曝光类型：'over'表示过度曝光，'under'表示欠曝光")

    args = parser.parse_args()
    process_images(args.input_folder, args.output_folder, args.exposure_type)


# # 使用示例
# input_folder = '/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset/sequences/04/image_2'  # 输入文件夹路径
# output_folder = '/workspace/mnt/storage/shihao/EventSSC/SemanticKITTI/kitti/dataset/over_expose/04/image_2'  # 输出文件夹路径
# process_images(input_folder, output_folder)
