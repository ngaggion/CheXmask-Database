import pandas as pd 
import cv2
import numpy as np
import matplotlib.pyplot as plt 

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

def getDenseMask(graph, imagesize = 1024):
    img = np.zeros([imagesize,imagesize])
    graph = graph.reshape(-1, 1, 2).astype('int')
    img = cv2.drawContours(img, [graph], -1, 255, -1)
    return img

path = "Datasets/CANDID/CANDID_RCA.csv"

df = pd.read_csv(path)

# Create an empty DataFrame
output = pd.DataFrame(columns=['ImageID', 'Dice RCA (Max)', 'Dice RCA (Mean)', 'Landmarks', 'Left Lung', 'Right Lung', 'Heart'])

# Iterate over the rows of the DataFrame
for index, row in df.iterrows():
    
    print(index)
    
    image_path = row['Image']
    dice_rca_max = row['Dice_RCA_Max']
    dice_rca_mean = row['Dice_RCA_Mean']
    
    if "Padchest" in image_path:
        landmark = image_path.replace('Images', 'Output').replace('.png', '.txt')
        id = image_path.split('/')[-1]
    elif "MIMIC" in image_path:
        landmark = image_path.replace('Images', 'Output').replace('.jpg', '.txt')
    elif "CANDID" in image_path:
        landmark = image_path.replace('Images', 'Output').replace('.jpg', '.txt')
        id = image_path.split('/')[-1]
    elif "VinBigData" in image_path:
        landmark = image_path.replace('pngs', 'Output').replace('.png', '.txt')
    elif "CheXpert" in image_path:
        landmark = image_path.replace('Preprocessed', 'Output').replace('.png', '.txt')
    elif "Chest8" in image_path:
        landmark = image_path.replace("ChestX-ray8", "Output").replace('images/','')[0:-3] + 'txt'
    
    data = np.loadtxt(landmark, delimiter=' ').astype('int')
    flattened_data = data.flatten()
    coordinates_str = ','.join(map(str, flattened_data))
    
    RL = getDenseMask(data[:44])
    LL = getDenseMask(data[44:94])
    H = getDenseMask(data[94:])
    
    RL_RLE = get_RLE_from_mask(RL)
    LL_RLE = get_RLE_from_mask(LL)
    H_RLE = get_RLE_from_mask(H)
    
    new_row = {
        'ImageID': id, 
        'Dice RCA (Max)': dice_rca_mean, 
        'Dice RCA (Mean)': dice_rca_max, 
        'Landmarks': coordinates_str, 
        'Right Lung': RL_RLE, 
        'Left Lung': LL_RLE, 
        'Heart': H_RLE
    }
    
    output = output.append(new_row, ignore_index=True)

output.to_csv("Annotations/CANDID-PTX.csv")
