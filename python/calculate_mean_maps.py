import numpy as np
import glob
import cv2
import tqdm

p = 'G:/TS_dataset/test/*/maps/*'

files = glob.glob(p)
n = 0
ims = 0
for file in tqdm.tqdm(files):
    n = n + 1

    im = cv2.imread(file, -1)
    im = im.astype(np.float64)

    ims = ims + im

mean_im = ims/n

mean_im = mean_im.astype(np.uint8)

cv2.imwrite(r'G:\TS_dataset\TS_mean_gt_maps.png', mean_im)