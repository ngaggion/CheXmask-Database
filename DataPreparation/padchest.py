import numpy as np
import os
import cv2 
import pandas as pd
import gc

def remove_padding(img):
    
    gray = 255*(img > 1) # To invert the text to white
    coords = cv2.findNonZero(gray) # Find all non-zero points (text)

    x, y, w, h = cv2.boundingRect(coords) # Find minimum spanning bounding box
    cropimg = img[y:y+h, x:x+w] # Crop the image - note we do this on the original image
    
    pad_left, pad_top, pad_right, pad_bottom = x, y, img.shape[1] - x - w, img.shape[0] - y - h
    
    return cropimg, pad_left, pad_top, pad_right, pad_bottom

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

df = pd.read_csv("padchest/PADCHEST_chest_x_ray_images_labels_160K_01.02.19_filtered.csv.gz")
new_csv = pd.DataFrame(columns=['filename', 'width', 'height', 'pad_left', 'pad_top', 'pad_right', 'pad_bottom'])       

images = df.ImageID.values

del df
gc.collect()

j = 0
n = len(images)

print("Total images: %s"%n)

errors = []


with open("padchest_padd_added.csv", '+w') as f:
    f.write("filename,height,width,pad_left_rem,pad_top_rem,pad_right_rem,pad_bottom_rem,pad_left,pad_top,pad_right,pad_bottom \n")
    
    with open("padchest_errors.csv", "+w") as errorf:
        for image in images:
            if j % 1000 == 0:
                print("Image %s of %s"%(j, n))
            j = j+1
            
            img_path = os.path.join("padchest/Images", image)
            out_path = os.path.join("padchest_preprocessed", image)
            
            if os.path.isfile(out_path):
                continue
                
            try:    
                img = cv2.imread(img_path, 0)
                h, w = img.shape[:2]
                
                # remove padding
                img2, pad_left, pad_top, pad_right, pad_bottom = remove_padding(img)
                reshaped, pad_left2, pad_top2, pad_right2, pad_bottom2 = pad_to_square_centered(img2)
                # scale to 1024x1024
                scaled = cv2.resize(reshaped, (1024, 1024))
                cv2.imwrite(out_path, scaled)
                
                del img
                del img2
                del scaled           
                
                gc.collect()
                
                outline = "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d \n"%(image, h, w, pad_left, pad_top, pad_right, pad_bottom, pad_left2, pad_top2, pad_right2, pad_bottom2)
                f.write(outline)
            except:
                errorf.write(image + " \n")
                print("Error with image %s"%image)
