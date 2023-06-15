import pandas as pd 
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

def getDenseMask(graph, h, w):
    img = np.zeros([h, w])
    graph = graph.reshape(-1, 1, 2).astype('int')
    img = cv2.drawContours(img, [graph], -1, 255, -1)
    return img


path = "../Annotations/Preprocessed/Padchest.csv"
path2 = ... # path to padding csv

df = pd.read_csv(path)
pads = pd.read_csv(path2)
pads.set_index('filename', inplace=True)

new_df = pd.DataFrame(columns=df.columns)

for index, row in df.iterrows():
    print(index)
    image_id = row["ImageID"]

    pad_row = pads.loc[image_id]

    height, width = pad_row["width"], pad_row["height"]

    # pad_left_rem	pad_top_rem	pad_right_rem	pad_bottom_rem
    pad_left_rem = pad_row["pad_left_rem"]
    pad_top_rem = pad_row["pad_top_rem"]
    pad_right_rem = pad_row["pad_right_rem"]
    pad_bottom_rem = pad_row["pad_bottom_rem"]

    height2 = height - pad_top_rem - pad_bottom_rem
    width2 = width - pad_left_rem - pad_right_rem

    landmarks = np.array(eval(row["Landmarks"])).reshape(-1, 2) / 1024
    max_shape = max(height2, width2)
    landmarks = landmarks * max_shape

    pad_left = pad_row["pad_left"]
    pad_top = pad_row["pad_top"]

    landmarks[:, 0] = landmarks[:, 0] - pad_left + pad_left_rem
    landmarks[:, 1] = landmarks[:, 1] - pad_top + pad_top_rem
    landmarks = np.round(landmarks).astype(int)

    RL = landmarks[:44]
    LL = landmarks[44:94]
    H = landmarks[94:]

    RL_ = getDenseMask(RL, height, width)
    LL_ = getDenseMask(LL, height, width)
    H_ = getDenseMask(H, height, width)

    RL_RLE = get_RLE_from_mask(RL_)
    LL_RLE = get_RLE_from_mask(LL_)
    H_RLE = get_RLE_from_mask(H_)

    # columns = image_id Dice RCA (Mean)	Dice RCA (Max)	Landmarks	Left Lung	Right Lung	Heart	Height	Width
        
    new_row = {
        "ImageID": row["ImageID"],
        "Dice RCA (Mean)": row["Dice RCA (Mean)"],
        "Dice RCA (Max)": row["Dice RCA (Max)"],
        "Landmarks": landmarks,
        "Left Lung": LL_RLE,
        "Right Lung": RL_RLE,
        "Heart": H_RLE,
        "Height": height,
        "Width": width
    }

    new_df = new_df.append(new_row, ignore_index=True)

new_df.to_csv("../Annotations/OriginalResolution/Padchest.csv", index=False)