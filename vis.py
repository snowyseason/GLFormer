import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import visualization as viz
from tqdm import tqdm

dataset = 'PN2'
cam_path = 'rslogs/PN2_conformer-s_cosrefine_cpk01/cos_repeat1'

for foldername, subfolders, filenames in os.walk(f'{cam_path}/scoremaps'):
    print(f'Start processing {subfolders}')
    for filename in tqdm(filenames):
        if filename.endswith('.npy'):
            filepath = f'{foldername}/{filename}'

            # 拼接路径获取图像
            part_flies = filepath.split('/')
            last_two_parts = '/'.join(part_flies[-2:])[:-4]
            image_path = os.path.join(f'datasets/{dataset}', last_two_parts)

            # 读取并处理图像和热力图
            img = cv2.resize(cv2.imread(image_path), (224, 224))
            heatmap = np.expand_dims(np.load(filepath), axis=2)

            # 核心去白边步骤1：关闭坐标轴
            plt.axis('off')

            # 绘制热力图（保持你的原始参数）
            viz.visualize_image_attr_multiple(
                heatmap,
                img,
                ['blended_heat_map'],
                ['positive'],
                show_colorbar=False,
                cmap='hot'
            )

            # 核心去白边步骤2：保存时裁剪白边（两个参数缺一不可）
            plt.savefig(
                f'{filepath[:-4]}',
                bbox_inches='tight',  # 裁剪周围空白
                pad_inches=0.0        # 裁剪后不留任何余量
            )

            # 关闭画布释放内存（循环中必备，避免卡顿）
            plt.close()
