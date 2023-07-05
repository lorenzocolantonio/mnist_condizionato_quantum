import numpy as np
import math
import time
import torch
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Modules.training_functions import *
from Modules.pennylane_functions import *
how_many=64
# if gpu available, set device to gpu
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using the GPU")
else:
    device = torch.device("cpu")
    print("WARNING: Could not find GPU, using the CPU")
T=5
classes=8
all=np.load(f'Data/dataset_ld_{ld_dim}_{9}.npy')
all=all[:how_many]
ld_dim=16
label=np.full((how_many),7)
for i in range(classes-1):
    x=np.load(f'Data/dataset_ld_{ld_dim}_{i}.npy')
    label_temp=np.full((how_many),i)
    

    all=np.concatenate((x[:how_many],all))
    label=np.concatenate((label_temp,label))
   
random_index = np.random.permutation(how_many*classes)
all=torch.tensor(all).to(device)
label=torch.tensor(label).to(device)
# Mischiare i campioni e le etichette utilizzando l'indice di permutazione casuale

#data_loader = torch.utils.data.DataLoader(mnist_images, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

qc_array=np.array([0])
min_array=np.array([0.05])
layer_array=np.array([0]) 
num_batch=classes
print(NUM_QUBITS)
print(T)
zero = torch.zeros(BATCH_SIZE, 2**NUM_QUBITS-ld_dim).to(device)

for layer_indx in range(len(layer_array)):
    n_layer=layer_array[layer_indx]
    for q_indx in range(len(qc_array)):
        qc=qc_array[q_indx]
        for min_indx in range(len(min_array)):
            min_b=min_array[min_indx]

            betas      = np.insert(np.linspace(10e-2,min_b, T), 0, 0)
            print(np.shape(betas))
            alphas     = 1 - betas
            alphas_bar = np.cumprod(alphas)
            pi         = math.pi
            betas      = torch.tensor(betas).float().to(device)
            alphas     = torch.tensor(alphas).float().to(device)
            alphas_bar = torch.tensor(alphas_bar).float().to(device)
            theta_1    = Variable(torch.rand((n_layer*3*NUM_QUBITS+n_layer*3*(NUM_QUBITS)), device = device), requires_grad=True)
            optimizer = torch.optim.Adam([theta_1], lr = LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = SCHEDULER_PATIENCE, gamma = SCHEDULER_GAMMA, verbose = False)
            trained_thetas_1 = []
            loss_history = []
            best_loss = 1e10

      
            random_index = np.random.permutation(how_many*classes)

                # Mischiare i campioni e le etichette utilizzando l'indice di permutazione casuale
            mnist_images = all[random_index]
            labels = label[random_index]
            target_batch=mnist_images[:256]
            fig, axs = plt.subplots(1, 6, figsize=(15, 3))
            axs[0].imshow(torch.abs(target_batch.cpu()[0]).reshape(2,8))
            
            for t_indx in range(T): #qua dovrebbe essere il numero di batch, ma visto che sono uguali al numero di classi metto il numero di clasii
                t=torch.full((BATCH_SIZE, ),t_indx, device=device)
                #t = torch.randint(0, T, size = (BATCH_SIZE, ), device=device)
                betas_batch = betas[t].to(device)
                alphas_batch=alphas_bar[t].to(device)

                # assemble input at t add noise (t+1)
                '''target_batch = assemble_input(images, t, alphas_bar,ld_dim ,device)
                target_batch = target_batch / torch.norm(target_batch, dim = 1).view(-1, 1)'''
                
                
            
                input_batch  = noise_step(target_batch, t+1, betas,ld_dim, device)
                input_batch  = input_batch / torch.norm(input_batch, dim = 1).view(-1, 1)
                

                    # Itera attraverso le immagini e i subplot corrispondenti
                
                
                axs[t_indx+1].imshow(torch.abs(input_batch.cpu()[0]).reshape(2,8))
                
                target_batch = input_batch

                #print(target_batch[0])
            plt.show()
            print(np.shape(target_batch))
            fig, axs = plt.subplots(1, 8, figsize=(12, 4), sharey=True)

            # Crea gli istogrammi per ogni colonna
            for i in range(8):
                axs[i].hist(torch.imag(target_batch).cpu()[:, i], bins=10)  # Modifica il numero di bin come desiderato

                # Aggiungi etichette
                axs[i].set_title(f'Colonna {i+1}')
                axs[i].set_xlabel('Valore')
                axs[i].set_ylabel('Frequenza')

            # Mostra il grafico
            plt.tight_layout()
            plt.show()
