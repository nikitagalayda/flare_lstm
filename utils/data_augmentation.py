import tensorflow as tf
from tensorflow.keras.layers import Concatenate
import numpy as np
from scipy.ndimage.interpolation import rotate

IMAGE_WIDTH, IMAGE_HEIGHT = 128, 128

def FlipImages(img, height=IMAGE_WIDTH, width=IMAGE_HEIGHT, channels=1):
    # flip_1 = np.fliplr(img)
    # shape = [height, width, channels]
    # x = tf.placeholder(dtype = tf.float32, shape = shape)
    x = tf.convert_to_tensor(img)
    flip_2 = tf.image.flip_up_down(x)
    flip_3 = tf.image.flip_left_right(x)
    flip_4 = tf.image.random_flip_up_down(x)
    flip_5 = tf.image.random_flip_left_right(x)
    
    return Concatenate(axis=0)([flip_2, flip_3, flip_4, flip_5])

# def FlipImages(images):
#     flipped_images = []
    
#     for im in images:
#         flipped_images.append(FlipImage(im))
    
#     return flipped_images

def RotateImages(img, height=IMAGE_WIDTH, width=IMAGE_HEIGHT, channels=1):
    # shape = [height, width, channels]
    # x = tf.placeholder(dtype = tf.float32, shape = shape)
    x = tf.convert_to_tensor(img)
    rot_90 = tf.image.rot90(x, k=1)
    rot_180 = tf.image.rot90(x, k=2)
    
    return Concatenate(axis=0)(rot_90, rot_180)

def translate(img, shift=20, direction='right', roll=True):
    assert direction in ['right', 'left', 'down', 'up'], 'Directions should be top|up|left|right'
    img = img.copy()
    if direction == 'right':
        right_slice = img[:, -shift:].copy()
        img[:, shift:] = img[:, :-shift]
        if roll:
            img[:,:shift] = np.fliplr(right_slice)
    if direction == 'left':
        left_slice = img[:, :shift].copy()
        img[:, :-shift] = img[:, shift:]
        if roll:
            img[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = img[-shift:, :].copy()
        img[shift:, :] = img[:-shift,:]
        if roll:
            img[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = img[:shift, :].copy()
        img[:-shift, :] = img[shift:, :]
        if roll:
            img[-shift:,:] = upper_slice
    return img

def random_crop(img, crop_size=(10, 10)):
    assert crop_size[0] <= img.shape[0] and crop_size[1] <= img.shape[1], "Crop size should be less than image size"
    img = img.copy()
    w, h = img.shape[:2]
    x, y = np.random.randint(h-crop_size[0]), np.random.randint(w-crop_size[1])
    img = img[y:y+crop_size[0], x:x+crop_size[1]]
    return img

def rotate_img(img, angle, bg_patch=(5,5)):
    assert len(img.shape) <= 3, "Incorrect image shape"
    rgb = len(img.shape) == 3
    if rgb:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1], :], axis=(0,1))
    else:
        bg_color = np.mean(img[:bg_patch[0], :bg_patch[1]])
    img = rotate(img, angle, reshape=False)
    mask = [img <= 0, np.any(img <= 0, axis=-1)][rgb]
    img[mask] = bg_color
    return img