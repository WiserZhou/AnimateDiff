import torch
import torchvision.transforms.functional as TF
import imageio
from PIL import Image
import os
from tqdm import tqdm
import numpy as np


def addNoise(in_path, out_path, sigma):
    # 读取图片并转为Tensor
    img_pil = Image.open(in_path)
    # TF.to_tensor将PIL图像转换成PyTorch张量；unsqueeze在原来的基础上增加了一个批次维度
    img_tensor = TF.to_tensor(img_pil).to(device).unsqueeze(0)

    # 添加噪声
    with torch.no_grad():
        means = torch.tensor(0.).to(device)
        noise = torch.normal(means, sigma.repeat(img_tensor.shape)).to(device)
        img_tensor += noise
        img_tensor.clamp(0., 255.)

    # 转回PIL Image对象并保存
    img_pil_noisy = TF.to_pil_image(img_tensor.squeeze().cpu(), mode='RGB')
    img_pil_noisy.save(out_path)


dir_name = r'D:\AnimateDiff\learn\_assets_'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for i in tqdm(range(30), desc="Adding Noise"):
    addNoise(os.path.join(dir_name, f'{i}.jpg'), os.path.join(dir_name, f'{i + 1}.jpg'),
             torch.tensor(0.005 * i, device=device))

# 不需要将图像数据放入tensor列表中再保存为gif，可以直接从文件读取并保存
images = []
for i in tqdm(range(30), desc="generate GIF"):
    images.append(np.array(Image.open(os.path.join(dir_name, f'{i + 1}.jpg'))))

imageio.mimsave(os.path.join(dir_name, 'sample.gif'), images, fps=15)
