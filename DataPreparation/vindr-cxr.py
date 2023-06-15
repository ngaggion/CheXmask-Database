import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

def read_xray(path, voi_lut = True, fix_monochrome = True):
    dicom = pydicom.read_file(path)
    
    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array
               
    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
        
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
        
    return data

import pathlib
import re

def get_dicom_fps(dicom_dir):
    dicom_dir = pathlib.Path(dicom_dir)
    dicom_fps = list(dicom_dir.glob('*/*.dicom'))
    # read all dicom files
    dicom_fps = [str(fp) for fp in dicom_fps]
    return dicom_fps

path = "VinBigData/original/"
outpath = "VinBigData/pngs/"

dicom_fps = get_dicom_fps(path)
print('dicom_fps', len(dicom_fps))

# saves metadata in a csv file
import pandas as pd
import cv2

def pad_to_square_centered(img):
    h, w = img.shape[:2]
    longest_edge = max(h, w)
    result = np.zeros((longest_edge, longest_edge), dtype=img.dtype)
    pad_h = (longest_edge - h) // 2
    pad_w = (longest_edge - w) // 2
    result[pad_h:pad_h + h, pad_w:pad_w + w] = img

    pad_left = pad_w
    pad_top = pad_h
    pad_right = longest_edge - w - pad_w
    pad_bottom = longest_edge - h - pad_h

    return result, pad_left, pad_top, pad_right, pad_bottom

new_csv = pd.DataFrame(columns=['filename', 'width', 'height', 'pad_left', 'pad_top', 'pad_right', 'pad_bottom'])

j = 0
for file in dicom_fps:
    # print progress
    j += 1
    if j % 100 == 0:
        print(j, '/', len(dicom_fps))

    data = read_xray(file)
    filename = file.replace(path, '')
    h, w = data.shape[:2]

    reshaped, pad_left, pad_top, pad_right, pad_bottom = pad_to_square_centered(data)
    
    # scale to 1024x1024
    scaled = cv2.resize(reshaped, (1024, 1024))

    # save image
    file_out = outpath + filename.replace('.dicom', '.png')
    cv2.imwrite(file_out, scaled)
    new_csv = new_csv.append({'filename': filename, 'width': w, 'height': h, 'pad_left': pad_left, 'pad_top': pad_top, 'pad_right': pad_right, 'pad_bottom': pad_bottom}, ignore_index=True)

save_csv = outpath + 'paddings.csv'
new_csv.to_csv(save_csv, index=False)