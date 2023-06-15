import os 

import torch
from skimage import io, transform

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2 

class LandmarksDataset(Dataset):
    def __init__(self, images, img_path, label_path, transform=None, heart = False):
        
        self.images = images
        self.img_path = img_path
        self.label_path = label_path
        
        self.RL_path = os.path.join(self.label_path, 'RL')
        self.LL_path = os.path.join(self.label_path, 'LL')
        
        if heart:
            self.H_path = os.path.join(self.label_path, 'H')
            
        self.heart = heart

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.images[idx]
        
        img_path = os.path.join(self.img_path, img_name)
                
        RL_path = os.path.join(self.RL_path, img_name.replace('.png', '.npy'))       
        LL_path = os.path.join(self.LL_path, img_name.replace('.png', '.npy'))
                         
        image = io.imread(img_path).astype('float') / 255.0
        image = np.expand_dims(image, axis=2)
                               
        RL = np.load(RL_path).astype('float')
        LL = np.load(LL_path).astype('float')
        
        if self.heart: 
            H_path = os.path.join(self.H_path, img_name.replace('.png', '.npy'))
            H = np.load(H_path).astype('float')
            landmarks = np.concatenate([RL, LL, H], axis = 0)
        else:
            landmarks = np.concatenate([RL, LL], axis = 0)
                
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample

    

class RandomScale(object):
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']       
        
        # Pongo limites para evitar que los landmarks salgan del contorno
        min_x = np.min(landmarks[:,0]) 
        max_x = np.max(landmarks[:,0])
        ancho = max_x - min_x
        
        min_y = np.min(landmarks[:,1])
        max_y = np.max(landmarks[:,1])
        alto = max_y - min_y
        
        max_var_x = 1024 / ancho 
        max_var_y = 1024 / alto
                
        min_var_x = 0.70
        min_var_y = 0.70
                                
        varx = np.random.uniform(min_var_x, max_var_x)
        vary = np.random.uniform(min_var_y, max_var_y)
                
        landmarks[:,0] = landmarks[:,0] * varx
        landmarks[:,1] = landmarks[:,1] * vary
        
        h, w = image.shape[:2]
        new_h = np.round(h * vary).astype('int')
        new_w = np.round(w * varx).astype('int')

        img = transform.resize(image, (new_h, new_w))
        
        # Cropeo o padeo aleatoriamente
        min_x = np.round(np.min(landmarks[:,0])).astype('int')
        max_x = np.round(np.max(landmarks[:,0])).astype('int')
        
        min_y = np.round(np.min(landmarks[:,1])).astype('int')
        max_y = np.round(np.max(landmarks[:,1])).astype('int')
        
        if new_h > 1024:
            rango = 1024 - (max_y - min_y)
            maxl0y = new_h - 1025
            
            if rango > 0 and min_y > 0:
                l0y = min_y - np.random.randint(0, min(rango, min_y))
                l0y = min(maxl0y, l0y)
            else:
                l0y = min_y
                
            l1y = l0y + 1024
            
            img = img[l0y:l1y,:]
            landmarks[:,1] -= l0y
            
        elif new_h < 1024:
            pad = h - new_h
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((p0, p1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,1] += p0
        
        if new_w > 1024:
            rango = 1024 - (max_x - min_x)
            maxl0x = new_w - 1025
            
            if rango > 0 and min_x > 0:
                l0x = min_x - np.random.randint(0, min(rango, min_x))
                l0x = min(maxl0x, l0x)
            else:
                l0x = min_x
            
            l1x = l0x + 1024
                
            img = img[:, l0x:l1x]
            landmarks[:,0] -= l0x
            
        elif new_w < 1024:
            pad = w - new_w
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((0, 0), (p0, p1), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,0] += p0
        
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            print('Original', [new_h,new_w])
            print('Salida', img.shape)
            raise Exception('Error')
            
        return {'image': img, 'landmarks': landmarks}
    
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
 
    # apply gamma correction using the lookup table
    return np.float32(cv2.LUT(image.astype('uint8'), table))

class AugColor(object):
    def __init__(self, gammaFactor):
        self.gammaf = gammaFactor

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        # Gamma
        gamma = np.random.uniform(1 - self.gammaf, 1 + self.gammaf / 2)
        
        image[:,:,0] = adjust_gamma(image[:,:,0] * 255, gamma) / 255
        
        # Adds a little noise
        image = image + np.random.normal(0, 1/128, image.shape)
        
        return {'image': image, 'landmarks': landmarks}

class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        
        angle = np.random.uniform(- self.angle, self.angle)
        
        # pad to keep all the image
        image = np.pad(image, ((250, 250), (250, 250), (0, 0)), mode='constant', constant_values=0)
        image = transform.rotate(image, angle)
        image = image[250:1024+250, 250:1024+250]

        centro = image.shape[0] / 2, image.shape[1] / 2
        
        landmarks -= centro
        
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        landmarks = np.dot(landmarks, R)
        
        landmarks += centro

        return {'image': image, 'landmarks': landmarks}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
                
        size = image.shape[0]
        image = image.transpose((2, 0, 1))
        landmarks = landmarks / size
        landmarks = np.clip(landmarks, 0, 1)
        
        # concatenate a vectors of 1 to the landmarks 
        landmarks = np.concatenate((landmarks, np.ones((landmarks.shape[0], 1))), axis=1)
        
        return {'image': torch.from_numpy(image).float(),
                'landmarks': torch.from_numpy(landmarks).float()}


class RandomScaleRotate(object):
    def __init__(self, angle, scale_min = 0.7):
        self.angle = angle
        self.scale_min = scale_min

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']       
        
        # Roto los landmarks

        angle = np.random.uniform(- self.angle, self.angle)
        
        # pad to keep all the image
        image = np.pad(image, ((250, 250), (250, 250), (0, 0)), mode='constant', constant_values=0)
        image = transform.rotate(image, angle)
        image = image[250:1024+250, 250:1024+250]

        centro = image.shape[0] / 2, image.shape[1] / 2
        
        landmarks -= centro
        
        theta = np.deg2rad(angle)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))
        
        landmarks = np.dot(landmarks, R)
        landmarks += centro

        landmarks = np.clip(landmarks, 0, 1024)

        # Pongo limites para evitar que los landmarks salgan del contorno
        min_x = np.min(landmarks[:,0]) 
        max_x = np.max(landmarks[:,0])
        ancho = max_x - min_x
        
        min_y = np.min(landmarks[:,1])
        max_y = np.max(landmarks[:,1])
        alto = max_y - min_y
        
        max_var_x = 1024 / ancho 
        max_var_y = 1024 / alto
                
        min_var_x = self.scale_min
        min_var_y = self.scale_min
                                
        varx = np.random.uniform(min_var_x, max_var_x)
        vary = np.random.uniform(min_var_y, max_var_y)
                
        landmarks[:,0] = landmarks[:,0] * varx
        landmarks[:,1] = landmarks[:,1] * vary
        
        h, w = image.shape[:2]
        new_h = np.round(h * vary).astype('int')
        new_w = np.round(w * varx).astype('int')

        img = transform.resize(image, (new_h, new_w))
        
        # Cropeo o padeo aleatoriamente
        min_x = np.round(np.min(landmarks[:,0])).astype('int')
        max_x = np.round(np.max(landmarks[:,0])).astype('int')
        
        min_y = np.round(np.min(landmarks[:,1])).astype('int')
        max_y = np.round(np.max(landmarks[:,1])).astype('int')
        
        if new_h > 1024:
            rango = 1024 - (max_y - min_y)
            maxl0y = new_h - 1025
            
            if rango > 0 and min_y > 0:
                l0y = min_y - np.random.randint(0, min(rango, min_y))
                l0y = min(maxl0y, l0y)
            else:
                l0y = min_y
                
            l1y = l0y + 1024
            
            img = img[l0y:l1y,:]
            landmarks[:,1] -= l0y
            
        elif new_h < 1024:
            pad = h - new_h
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((p0, p1), (0, 0), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,1] += p0
        
        if new_w > 1024:
            rango = 1024 - (max_x - min_x)
            maxl0x = new_w - 1025
            
            if rango > 0 and min_x > 0:
                l0x = min_x - np.random.randint(0, min(rango, min_x))
                l0x = min(maxl0x, l0x)
            else:
                l0x = min_x
            
            l1x = l0x + 1024
                
            img = img[:, l0x:l1x]
            landmarks[:,0] -= l0x
            
        elif new_w < 1024:
            pad = w - new_w
            p0 = np.random.randint(np.floor(pad/4), np.ceil(3*pad/4))
            p1 = pad - p0
            
            img = np.pad(img, ((0, 0), (p0, p1), (0, 0)), mode='constant', constant_values=0)
            landmarks[:,0] += p0
        
        if img.shape[0] != 1024 or img.shape[1] != 1024:
            print('Original', [new_h,new_w])
            print('Salida', img.shape)
            raise Exception('Error')
            
        return {'image': img, 'landmarks': landmarks}