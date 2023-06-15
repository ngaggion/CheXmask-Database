import pathlib
import re
import numpy as np

def load_paths(path):
    path = pathlib.Path(path)
    paths = list(path.glob('*'))
    return [str(fp) for fp in paths]

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

path = "CheXpert-v1 0/Original"
outpath = "CheXpert-v1 0/Preprocessed"


# saves metadata in a csv file
import cv2
import os

with open("CheXpert-v1 0/Preprocessed/paddings.csv", '+w') as f:
    f.write("filename,width,height,pad_left,pad_top,pad_right,pad_bottom \n")
    
    j = 0
    for folder in load_paths(path):
        if '.csv' in folder:
            continue
        
        out_folder = outpath + '/' + folder.split('/')[-1]
        # if out_folder does not exist, create it
        try:
            os.mkdir(out_folder)
        except:
            pass
        
        n = len(load_paths(folder))
        j = 0
        for patient in load_paths(folder):
            if j%100 == 0:
                print(j, 'of', n)
            j += 1
                    
            out_patient_folder = out_folder + '/' + patient.split('/')[-1]
            
            try:
                os.mkdir(out_patient_folder)
            except:
                pass
            
            for study in load_paths(patient):
                out_study_folder = out_patient_folder + '/' + study.split('/')[-1]
                
                try:
                    os.mkdir(out_study_folder)
                except:
                    pass
                
                imgpath = study + '/view1_frontal.jpg'
                
                # save image
                file_out = out_study_folder + '/view1_frontal.png'
                    
                if os.path.isfile(imgpath):
            
                    data = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
                    h, w = data.shape[:2]

                    reshaped, pad_left, pad_top, pad_right, pad_bottom = pad_to_square_centered(data)
                    
                    # scale to 1024x1024
                    scaled = cv2.resize(reshaped, (1024, 1024))

                    cv2.imwrite(file_out, scaled)
                    
                    outline = "%s,%d,%d,%d,%d,%d,%d \n"%(file_out.replace(outpath, ''), h, w, pad_left, pad_top, pad_right, pad_bottom)
                    f.write(outline)