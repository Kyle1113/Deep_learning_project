import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# 이미지와 마스크 파일 경로 설정
image_dir = '.data/image'
mask_dir = '.data/mask'

# 이미지와 마스크 파일 목록 가져오기
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.npy')])

# 데이터 로드 및 전처리 함수 정의
def load_image(image_path):
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = img / 255.0  # Normalize to [0, 1]
    return img

def load_mask(mask_path):
    mask = np.load(mask_path)
    mask = np.resize(mask, (img_height, img_width, 1))
    mask = mask / 255.0  # Normalize to [0, 1] if necessary
    return mask

def load_data(image_files, mask_files):
    images = [load_image(os.path.join(image_dir, f)) for f in image_files]
    masks = [load_mask(os.path.join(mask_dir, f)) for f in mask_files]
    return np.array(images), np.array(masks)

# 데이터셋 생성
images, masks = load_data(image_files, mask_files)
dataset = tf.data.Dataset.from_tensor_slices((images, masks))
dataset = dataset.shuffle(len(images)).batch(batch_size).prefetch(tf.data.AUTOTUNE)
