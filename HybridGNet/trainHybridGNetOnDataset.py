import os
import torch
import torch.nn.functional as F
import argparse
import random

from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import scipy.sparse as sp
import numpy as np
import pandas as pd

from utils.utils import scipy_to_torch_sparse, genMatrixesLungs, genMatrixesLungsHeart, CrossVal
from utils.dataset_for_train import LandmarksDataset, ToTensor

from models.modelUtils import Pool
from models.HybridGNet2IGSC import Hybrid 
import time

def trainer(train_dataset, val_dataset, model, config):
    torch.manual_seed(420)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = model.to(device)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = config['batch_size'], 
                                               shuffle = True, num_workers = 8, drop_last = True)
    
    optimizer = torch.optim.Adam(params = model.parameters(), lr = config['lr'], weight_decay = config['weight_decay'])

    train_loss_avg = []
    train_seg_loss_avg = []
    train_kld_loss_avg = []
    val_loss_avg = []
    
    tensorboard = "Training/RCA_Dataset"
        
    folder = os.path.join(tensorboard, config['name'])

    try:
        os.mkdir(folder)
    except:
        pass 

    writer = SummaryWriter(log_dir = folder)  

    best = 1e12
    
    print('Training ...')
        
    scheduler = StepLR(optimizer, step_size=config['stepsize'], gamma=config['gamma'])
    pool = Pool()
    
    iterator = train_loader.__iter__()
    len_train = len(train_loader)
    count = 0
    
    for epoch in range(config['epochs']):        
        model.train()

        train_loss_avg.append(0)
        train_seg_loss_avg.append(0)
        train_kld_loss_avg.append(0)
        num_batches = 0
        
        t_1 = time.time()
        
        for j in range(0, 500):
            
            if count % len_train == 0:
                iterator = train_loader.__iter__()
                count = 0
            
            sample_1 = iterator.__next__()
            count += 1
            
            image_1, target_1 = sample_1['image'].to(device), sample_1['landmarks'].to(device)
            
            out_1 = model(image_1)
            
            optimizer.zero_grad()
                        
            target_down_1 = pool(target_1, model.downsample_matrices[0])
            ts = target_1.shape[1]
            ts_d= target_down_1.shape[1]

            out, pre1, pre2 = out_1
            
            pre1loss = F.mse_loss(pre1[:,:ts_d,:], target_down_1[:,:,:2])
            pre2loss = F.mse_loss(pre2[:,:ts,:], target_1[:,:,:2])
            outloss = F.mse_loss(out[:,:ts,:], target_1[:,:,:2]) 
            
            loss = outloss + pre1loss + pre2loss
            
            # KLD loss
            kld_loss = -0.5 * torch.mean(torch.mean(1 + model.log_var - model.mu ** 2 - model.log_var.exp(), dim=1), dim=0)
           
            # Total loss
            loss = loss + 1e-5 * kld_loss 

            train_seg_loss_avg[-1] += outloss.item() 
            train_kld_loss_avg[-1] += kld_loss.item() 
            train_loss_avg[-1] += loss.item()

            loss.backward()

            # one step of the optmizer (using the gradients from backpropagation)
            optimizer.step()

            num_batches += 1
            
            if j % 100 == 0:
                t_2 = time.time()
                print('Epoch [%d / %d] Batch [%d / %d] train segmentation error: %f' % (epoch+1, config['epochs'], j, 500, outloss.item()*1024*1024))
                print('Time since epoch beggining: %f' % (t_2 - t_1))
                
        train_loss_avg[-1] /= num_batches
        train_seg_loss_avg[-1] /= num_batches
        train_kld_loss_avg[-1] /= num_batches

        print('Epoch [%d / %d] train average segmentation error: %f' % (epoch+1, config['epochs'], train_seg_loss_avg[-1]*1024*1024))
        print('Epoch [%d / %d] train average kld error: %f' % (epoch+1, config['epochs'], train_kld_loss_avg[-1]*1024*1024))
        
        t_2 = time.time()
        print('Total epoch time: %f' % (t_2 - t_1))
        
        num_batches = 0
        
        writer.add_scalar('Train/Loss', train_loss_avg[-1], epoch)
        writer.add_scalar('Train/Seg MSE', train_seg_loss_avg[-1] * 1024 * 1024, epoch)
        writer.add_scalar('Train/KLD', train_kld_loss_avg[-1], epoch)
        
        scheduler.step()
    
        torch.save(model.state_dict(), os.path.join(folder, "final.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model", default = "HybridGNet", type=str)    
    parser.add_argument("--epochs", default = 200, type = int)
    parser.add_argument("--lr", default = 1e-4, type = float)
    parser.add_argument("--stepsize", default = 1, type = int)
    parser.add_argument("--gamma", default = 0.99, type = float)
    
    ## 5-fold Cross validation fold
    parser.add_argument("--fold", default = 1, type = int)
    
    # Number of filters at low resolution for HybridGNet
    parser.add_argument("--f", default = 32, type=int)
                    
    parser.add_argument("--dataset", default = "CANDID", type=str)
    
    config = parser.parse_args()
    config = vars(config)

    print('Organs: Lungs and Heart')
    A, AD, D, U = genMatrixesLungsHeart()
    
    if config['dataset'] == "CANDID-PTX":
        df = pd.read_csv("../Annotations/CANDID-PTX.csv")
    elif config['dataset'] == "ChestX-Ray8":
        df = pd.read_csv('../Annotations/ChestX-Ray8.csv')
    elif config['dataset'] == "CheXpert":
        df = pd.read_csv('..-Annotations/CheXpert.csv')
    elif config['dataset'] == "MIMIC-CXR-JPG":
        df = pd.read_csv('../Annotations/MIMIC-CXR-JPG.csv')
    elif config['dataset'] == "Padchest":
        df = pd.read_csv('../Annotations/Padchest.csv')
    elif config['dataset'] == "VinDr-CXR":
        df = pd.read_csv('../Annotations/VinDr-CXR.csv')
    else:
        raise ValueError("Dataset not supported, please choose between CANDID-PTX, ChestX-Ray8, CheXpert, MIMIC-CXR-JPG, Padchest or VinDr-CXR")
        
    config['name'] = config['dataset'] 
    
    images = df.iloc[:,0].values
    
    # SET UP THE LANDMARK DATASET TO INCORPORATE THE PATH TO YOUR IMAGES
    train_dataset = LandmarksDataset(images=images,
                                     transform = ToTensor()
                                     )

    val_dataset = None                                                    
 
    config['latents'] = 64
    config['batch_size'] = 4
    config['val_batch_size'] = 1
    config['weight_decay'] = 1e-5
    config['inputsize'] = 1024
    
    config['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print('Model: HybrigGNet')
    
    f = int(config['f'])
    print(f, 'filters')
    config['filters'] = [2, f, f, f, f//2, f//2, f//2]

    N1 = A.shape[0]
    N2 = AD.shape[0]

    A = sp.csc_matrix(A).tocoo()
    AD = sp.csc_matrix(AD).tocoo()
    D = sp.csc_matrix(D).tocoo()
    U = sp.csc_matrix(U).tocoo()

    D_ = [D.copy()]
    U_ = [U.copy()]

    config['n_nodes'] = [N1, N1, N1, N2, N2, N2]
    A_ = [A.copy(), A.copy(), A.copy(), AD.copy(), AD.copy(), AD.copy()]
    A_t, D_t, U_t = ([scipy_to_torch_sparse(x).to('cuda:0') for x in X] for X in (A_, D_, U_))

    model = Hybrid(config, D_t, U_t, A_t)
        
    trainer(train_dataset, val_dataset, model, config)