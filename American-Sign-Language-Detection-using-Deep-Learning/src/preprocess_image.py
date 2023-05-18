import pandas as pd
import os
import cv2
import random
import albumentations 
import numpy as np
import argparse

from imutils import paths
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--num-images', default=1000, type=int,
    help='number of images to preprocess for each category')
args = vars(parser.parse_args())

print(f"Preprocessing {args['num_images']} from each category...")

image_paths = list(paths.list_images('C:/Users/Kartheek. B/Desktop/DL_project/American-Sign-Language-Detection-using-Deep-Learning/input/asl_alphabet_test/asl_alphabet_test'))
dir_paths = os.listdir('C:/Users/Kartheek. B/Desktop/DL_project/American-Sign-Language-Detection-using-Deep-Learning/input/asl_alphabet_train')
dir_paths.sort()

root_path = 'C:/Users/Kartheek. B/Desktop/DL_project/American-Sign-Language-Detection-using-Deep-Learning/input/asl_alphabet_train'

for idx, dir_path  in tqdm(enumerate(dir_paths), total=len(dir_paths)):
    all_images = os.listdir(f"{root_path}/{dir_path}")
    os.makedirs(f"../input/preprocessed_image/{dir_path}", exist_ok=True)
    for i in range(args['num_images']): 
        rand_id = (random.randint(0, 2999))
        image = cv2.imread(f"{root_path}/{dir_path}/{all_images[rand_id]}")
        image = cv2.resize(image, (224, 224))

        cv2.imwrite(f"../input/preprocessed_image/{dir_path}/{dir_path}{i}.jpg", image)

print('DONE')