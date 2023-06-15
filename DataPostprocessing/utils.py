import numpy as np 
import cv2

def get_RLE_from_mask(mask):
    mask = (mask / 255).astype(int)
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def get_mask_from_RLE(rle, height, width):
    runs = np.array([int(x) for x in rle.split()])
    starts = runs[::2]
    lengths = runs[1::2]

    mask = np.zeros((height * width), dtype=np.uint8)

    for start, length in zip(starts, lengths):
        start -= 1  
        end = start + length
        mask[start:end] = 255

    mask = mask.reshape((height, width))
    
    return mask

def getDenseMaskFromLandmarks(landmarks, imagesize = 1024):
    img = np.zeros([imagesize,imagesize])
    landmarks = landmarks.reshape(-1, 1, 2).astype('int')
    img = cv2.drawContours(img, [landmarks], -1, 255, -1)
    return img