import os
from concurrent.futures import ThreadPoolExecutor

import cv2


def fix_png_image(file_path):
    # img = Image.open(file_path)
    # img = img.convert('RGB')
    # img = img.resize((224, 224))
    # img.save(file_path, 'PNG', icc_profile=None)
    try:
        img = cv2.imread(file_path);
        img = cv2.resize(img, (224, 224))
        cv2.imwrite(file_path, img)
    except:
        print("~~~",file_path)


def process_directory(directory, num_threads=4):
    def process_file(file_path):
        try:
            fix_png_image(file_path)
            print(f"Fixed: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    png_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                file_path = os.path.join(root, file)
                png_files.append(file_path)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_file, png_files)


if __name__ == "__main__":
    # 请将此路径替换为你需要处理的文件夹路径
    directory = '/data/giraffe/0_FSL/data/inat2017_84x84'
    # 设置线程数量
    num_threads = 36
    process_directory(directory, num_threads)
