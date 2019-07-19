# USAGE
# python augmentation_demo.py --image jemma.png --output output

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import os
from imutils import paths
import cv2

def resize_pad(im,desired_size):
    old_size = im.shape[:2] # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    return new_im

#Current working directory
inDir     = r'C:\Users\somd7w\Desktop\Airway_Project\trainingdata_0.8'
outDir    = r'C:\Users\somd7w\Desktop\Airway_Project\augumenteddata_0.8'

# Determine number of classes
classes = os.listdir(inDir)
desired_size = 512
samples_per_img = 10
for group in classes:
    if not isdir(join(outDir,group)):
        os.makedirs(join(outDir,group))
    filelist = list(paths.list_images(join(inDir,group)))
    for f,file in enumerate(filelist):
        image = cv2.imread(file)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = resize_pad(image,desired_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.15,
	        height_shift_range=0.1, shear_range=0.1, zoom_range=0.15,
	        horizontal_flip=True, fill_mode="nearest", vertical_flip=True)
        total = 0
        # construct the actual Python generator
        print("[INFO] generating images...")
        fname = file.split(os.path.sep)[-1]
        imageGen = aug.flow(image, batch_size=1, save_to_dir=join(outDir,group),
	                save_prefix=fname, save_format="tif")
        # loop over examples from our image data augmentation generator
        for image in imageGen:
            total += 1
            if total == samples_per_img:
                break

