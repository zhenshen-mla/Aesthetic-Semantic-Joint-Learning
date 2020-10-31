import os
import numpy as np
import random

path_high_animal = '/home/liuxiangfei/PhotoQualityDataset/HighQuality/animal/'
path_high_architecture = '/home/liuxiangfei/PhotoQualityDataset/HighQuality/architecture/'
path_high_human = '/home/liuxiangfei/PhotoQualityDataset/HighQuality/human/'
path_high_landscape = '/home/liuxiangfei/PhotoQualityDataset/HighQuality/landscape/'
path_high_night = '/home/liuxiangfei/PhotoQualityDataset/HighQuality/night/'
path_high_plant = '/home/liuxiangfei/PhotoQualityDataset/HighQuality/plant/'
path_high_static = '/home/liuxiangfei/PhotoQualityDataset/HighQuality/static/'

path_low_animal = '/home/liuxiangfei/PhotoQualityDataset/LowQuality/animal/'
path_low_architecture = '/home/liuxiangfei/PhotoQualityDataset/LowQuality/architecture/'
path_low_human = '/home/liuxiangfei/PhotoQualityDataset/LowQuality/human/'
path_low_landscape = '/home/liuxiangfei/PhotoQualityDataset/LowQuality/landscape/'
path_low_night = '/home/liuxiangfei/PhotoQualityDataset/LowQuality/night/'
path_low_plant = '/home/liuxiangfei/PhotoQualityDataset/LowQuality/plant/'
path_low_static = '/home/liuxiangfei/PhotoQualityDataset/LowQuality/static/'

l_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
l_path = []
l_aes = []
l_seg = []

# high quality
for file in os.listdir(path_high_animal):
    filepath = os.path.join(path_high_animal, file)
    l_count[0] += 1
    l_path.append(filepath)
    l_aes.append(1)
    l_seg.append(0)
for file in os.listdir(path_high_architecture):
    filepath = os.path.join(path_high_architecture, file)
    l_count[1] += 1
    l_path.append(filepath)
    l_aes.append(1)
    l_seg.append(1)
for file in os.listdir(path_high_human):
    filepath = os.path.join(path_high_human, file)
    l_count[2] += 1
    l_path.append(filepath)
    l_aes.append(1)
    l_seg.append(2)
for file in os.listdir(path_high_landscape):
    filepath = os.path.join(path_high_landscape, file)
    l_count[3] += 1
    l_path.append(filepath)
    l_aes.append(1)
    l_seg.append(3)
for file in os.listdir(path_high_night):
    filepath = os.path.join(path_high_night, file)
    l_count[4] += 1
    l_path.append(filepath)
    l_aes.append(1)
    l_seg.append(4)
for file in os.listdir(path_high_plant):
    filepath = os.path.join(path_high_plant, file)
    l_count[5] += 1
    l_path.append(filepath)
    l_aes.append(1)
    l_seg.append(5)
for file in os.listdir(path_high_static):
    filepath = os.path.join(path_high_static, file)
    l_count[6] += 1
    l_path.append(filepath)
    l_aes.append(1)
    l_seg.append(6)


# low quality
for file in os.listdir(path_low_animal):
    filepath = os.path.join(path_low_animal, file)
    l_count[7] += 1
    l_path.append(filepath)
    l_aes.append(0)
    l_seg.append(0)
for file in os.listdir(path_low_architecture):
    filepath = os.path.join(path_low_architecture, file)
    l_count[8] += 1
    l_path.append(filepath)
    l_aes.append(0)
    l_seg.append(1)
for file in os.listdir(path_low_human):
    filepath = os.path.join(path_low_human, file)
    l_count[9] += 1
    l_path.append(filepath)
    l_aes.append(0)
    l_seg.append(2)
for file in os.listdir(path_low_landscape):
    filepath = os.path.join(path_low_landscape, file)
    l_count[10] += 1
    l_path.append(filepath)
    l_aes.append(0)
    l_seg.append(3)
for file in os.listdir(path_low_night):
    filepath = os.path.join(path_low_night, file)
    l_count[11] += 1
    l_path.append(filepath)
    l_aes.append(0)
    l_seg.append(4)
for file in os.listdir(path_low_plant):
    filepath = os.path.join(path_low_plant, file)
    l_count[12] += 1
    l_path.append(filepath)
    l_aes.append(0)
    l_seg.append(5)
for file in os.listdir(path_low_static):
    filepath = os.path.join(path_low_static, file)
    l_count[13] += 1
    l_path.append(filepath)
    l_aes.append(0)
    l_seg.append(6)

random.seed(702)
random.shuffle(l_path)
random.seed(702)
random.shuffle(l_aes)
random.seed(702)
random.shuffle(l_seg)

print(l_count)  # [953, 595, 678, 820, 353, 594, 531, 2292, 1290, 2470, 1950, 1356, 1803, 2005]
print(len(l_path), len(l_aes), len(l_seg))  # 17690 17690 17690
print(l_aes.count(0), l_aes.count(1))  # 13166 4524
print(l_seg.count(0), l_seg.count(1), l_seg.count(2), l_seg.count(3), l_seg.count(4), l_seg.count(5), l_seg.count(6))  # 3245 1885 3148 2770 1709 2397 2536

np.save('list_index_HK.npy', np.array(l_path))
np.save('list_aes_HK.npy', np.array(l_aes))
np.save('list_seg_HK.npy', np.array(l_seg))



