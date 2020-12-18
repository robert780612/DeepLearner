import os
from glob import glob
from PIL import Image

IMAGE_PATH = '/share/data/diabetic-retinopathy-detection/train'
IMAGE_PATH_OUT = '/share/data/diabetic-retinopathy-detection/train_resized'
images = glob(os.path.join(IMAGE_PATH, '*.jpeg'))


for i, im_path in enumerate(images):
    im_name = im_path.split('/')[-1]
    image = Image.open(im_path)
    image = image.resize((288, 288), resample=Image.BILINEAR)
    image.save(os.path.join(IMAGE_PATH_OUT, im_name), "jpeg")

