import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def draw_organ(ax, array, color = 'b'):
    N = array.shape[0]
    for i in range(0, N):
        x, y = array[i,:]
        circ = plt.Circle((x, y), radius=3, color=color, fill = True)
        ax.add_patch(circ)
    return

def draw_lines(ax, array, color = 'b'):
    N = array.shape[0]
    for i in range(0, N):
        x1, y1 = array[i-1,:]
        x2, y2 = array[i,:]
        ax.plot([x1, x2], [y1, y2], color=color, linestyle='-', linewidth=1)
    return

def drawOrgans(RL, LL, H = None, img =  None, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    
    if img is not None:
        plt.imshow(img, cmap='gray')
    else:
        img = np.zeros([1024, 1024])
        plt.imshow(img)
    
    plt.axis('off')
    
    draw_lines(ax, RL, 'r')
    draw_lines(ax, LL, 'g')
    
    draw_organ(ax, RL, 'r')
    draw_organ(ax, LL, 'g')
    
    if H is not None:
        draw_lines(ax, H, 'y')
        draw_organ(ax, H, 'y')

    return

import scipy.sparse as sp
import torch

def scipy_to_torch_sparse(scp_matrix):
    values = scp_matrix.data
    indices = np.vstack((scp_matrix.row, scp_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = scp_matrix.shape

    sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))
    return sparse_tensor

## Adjacency Matrix
def mOrgan(N):
    sub = np.zeros([N, N])
    for i in range(0, N):
        sub[i, i-1] = 1
        sub[i, (i+1)%N] = 1
    return sub

## Downsampling Matrix
def mOrganD(N):
    N2 = int(np.ceil(N/2))
    sub = np.zeros([N2, N])
    
    for i in range(0, N2):
        if (2*i+1) == N:
            sub[i, 2*i] = 1
        else:
            sub[i, 2*i] = 1/2
            sub[i, 2*i+1] = 1/2
            
    return sub

## Upsampling Matrix
def mOrganU(N):
    N2 = int(np.ceil(N/2))
    sub = np.zeros([N, N2])
    
    for i in range(0, N):
        if i % 2 == 0:
            sub[i, i//2] = 1
        else:
            sub[i, i//2] = 1/2
            sub[i, (i//2 + 1) % N2] = 1/2
            
    return sub

## Generating Matrixes for every organ
def genMatrixesLungs():       
    RLUNG = 44
    LLUNG = 50
    
    Asub1 = mOrgan(RLUNG)
    Asub2 = mOrgan(LLUNG)
    
    ADsub1 = mOrgan(int(np.ceil(RLUNG / 2)))
    ADsub2 = mOrgan(int(np.ceil(LLUNG / 2)))
                    
    Dsub1 = mOrganD(RLUNG)
    Dsub2 = mOrganD(LLUNG)
    
    Usub1 = mOrganU(RLUNG)
    Usub2 = mOrganU(LLUNG)
        
    p1 = RLUNG
    p2 = p1 + LLUNG
    
    p1_ = int(np.ceil(RLUNG / 2))
    p2_ = p1_ + int(np.ceil(LLUNG / 2))
    
    A = np.zeros([p2, p2])
    
    A[:p1, :p1] = Asub1
    A[p1:p2, p1:p2] = Asub2
    
    AD = np.zeros([p2_, p2_])
    
    AD[:p1_, :p1_] = ADsub1
    AD[p1_:p2_, p1_:p2_] = ADsub2
   
    D = np.zeros([p2_, p2])
    
    D[:p1_, :p1] = Dsub1
    D[p1_:p2_, p1:p2] = Dsub2
    
    U = np.zeros([p2, p2_])
    
    U[:p1, :p1_] = Usub1
    U[p1:p2, p1_:p2_] = Usub2
    
    return A, AD, D, U


def genMatrixesLungsHeart():       
    RLUNG = 44
    LLUNG = 50
    HEART = 26
    
    Asub1 = mOrgan(RLUNG)
    Asub2 = mOrgan(LLUNG)
    Asub3 = mOrgan(HEART)
    
    ADsub1 = mOrgan(int(np.ceil(RLUNG / 2)))
    ADsub2 = mOrgan(int(np.ceil(LLUNG / 2)))
    ADsub3 = mOrgan(int(np.ceil(HEART / 2)))
                    
    Dsub1 = mOrganD(RLUNG)
    Dsub2 = mOrganD(LLUNG)
    Dsub3 = mOrganD(HEART)
    
    Usub1 = mOrganU(RLUNG)
    Usub2 = mOrganU(LLUNG)
    Usub3 = mOrganU(HEART)
        
    p1 = RLUNG
    p2 = p1 + LLUNG
    p3 = p2 + HEART
    
    p1_ = int(np.ceil(RLUNG / 2))
    p2_ = p1_ + int(np.ceil(LLUNG / 2))
    p3_ = p2_ + int(np.ceil(HEART / 2))
    
    A = np.zeros([p3, p3])
    
    A[:p1, :p1] = Asub1
    A[p1:p2, p1:p2] = Asub2
    A[p2:p3, p2:p3] = Asub3
    
    AD = np.zeros([p3_, p3_])
    
    AD[:p1_, :p1_] = ADsub1
    AD[p1_:p2_, p1_:p2_] = ADsub2
    AD[p2_:p3_, p2_:p3_] = ADsub3
   
    D = np.zeros([p3_, p3])
    
    D[:p1_, :p1] = Dsub1
    D[p1_:p2_, p1:p2] = Dsub2
    D[p2_:p3_, p2:p3] = Dsub3
    
    U = np.zeros([p3, p3_])
    
    U[:p1, :p1_] = Usub1
    U[p1:p2, p1_:p2_] = Usub2
    U[p2:p3, p2_:p3_] = Usub3
    
    return A, AD, D, U


def CrossVal(all_files, iFold, k = 5):
    #Performs 5-Fold-CrossValidation
    
    total = len(all_files)
    val = int(total/k)
    
    indices = list(range(total))
    
    train_indices = indices[0:(iFold-1)*val] + indices[iFold*val:]
    val_indices = indices[(iFold-1)*val:iFold*val]

    train_paths = [all_files[i] for i in train_indices]
    val_paths = [all_files[i] for i in val_indices]
    
    return train_paths, val_paths