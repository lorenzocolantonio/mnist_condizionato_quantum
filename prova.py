import numpy as np
import math
import time
import torch
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
from Modules.training_functions import *
from Modules.pennylane_functions import *

# if gpu available, set device to gpu
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Using the GPU")
else:
    device = torch.device("cpu")
    print("WARNING: Could not find GPU, using the CPU")
T=5
classes=10
all=np.load(f'Data/dataset_ld_{ld_dim}_{9}.npy')
all=all[:256]

label=np.full((256),9)
for i in range(classes-1):
    x=np.load(f'Data/dataset_ld_{ld_dim}_{i}.npy')
    label_temp=np.full((256),i)
    

    all=np.concatenate((x[:256],all))
    label=np.concatenate((label_temp,label))
   
random_index = np.random.permutation(256*classes)
all=torch.tensor(all).to(device)
label=torch.tensor(label).to(device)
# Mischiare i campioni e le etichette utilizzando l'indice di permutazione casuale

#data_loader = torch.utils.data.DataLoader(mnist_images, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

qc_array=np.array([0,4,6,7])
min_array=np.array([0.05,0.01,0.005])
layer_array=np.array([10,20,50]) 
num_batch=2
print(NUM_QUBITS)
print(T)
zero = torch.zeros(BATCH_SIZE, 2**NUM_QUBITS-ld_dim).to(device)

for layer_indx in range(len(layer_array)):
    n_layer=layer_array[layer_indx]
    for q_indx in range(len(qc_array)):
        qc=qc_array[q_indx]
        for min_indx in range(len(min_array)):
            min_b=min_array[min_indx]

            betas      = np.insert(np.linspace(10e-8,min_b, T), 0, 0)
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

            for epoch in range(NUM_EPOCHS):
                print(epoch)

                t0 = time.time()
                num_batch=classes
                tot_loss=0
                random_index = np.random.permutation(256*classes)

                # Mischiare i campioni e le etichette utilizzando l'indice di permutazione casuale
                mnist_images = all[random_index]
                labels = label[random_index]
                
                for batch_indx in range(classes): #qua dovrebbe essere il numero di batch, ma visto che sono uguali al numero di classi metto il numero di clasii
                    loss_batch=0
                    image_batch=mnist_images[BATCH_SIZE*batch_indx:BATCH_SIZE*(batch_indx+1)]
                    label_batch=labels[BATCH_SIZE*batch_indx:BATCH_SIZE*(batch_indx+1)]
                    
                    
                    t = torch.randint(0, T, size = (BATCH_SIZE, ), device=device)
                    betas_batch = betas[t].to(device)
                    alphas_batch=alphas_bar[t].to(device)

                    # assemble input at t add noise (t+1)
                    target_batch = assemble_input(image_batch, t, alphas_bar,ld_dim ,device)
                    
                
                    input_batch  = noise_step(target_batch, t+1, betas,ld_dim, device)
                    target_batch = target_batch / torch.norm(target_batch, dim = 1).view(-1, 1)
                    input_batch  = input_batch / torch.norm(input_batch, dim = 1).view(-1, 1)
                    for j in range(classes):
                        
                        indici_classe = (label_batch == j).nonzero().squeeze()
                        
                        img_batch_temp=input_batch[indici_classe]
                        target_batch_temp=input_batch[indici_classe]
                        
                        zero_input = torch.zeros((len(img_batch_temp), 2**NUM_QUBITS), dtype=torch.complex64)
                        zero_target = torch.zeros((len(img_batch_temp), 2**NUM_QUBITS), dtype=torch.complex64)
                        

                        zero_input[:,j*ld_dim:(j+1)*ld_dim]=img_batch_temp
                        
                        zero_target[:,j*ld_dim:(j+1)*ld_dim]=target_batch_temp
                        
                        
                        loss_label = loss_fn_aq(qc,theta_1,n_layer, zero_input, zero_target,j)
                        #if j==9: print(f'loss{loss_label}, label={j}')
                        loss_batch+=loss_label/classes
                        
                    
                    
                    tot_loss+=loss_batch.item()/num_batch
                    loss_batch.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    
  


                trained_thetas_1.append(theta_1.cpu().clone().detach().numpy())

                loss_history.append(tot_loss)
                if tot_loss< best_loss:
                    best_loss=tot_loss

                # implement learning rate scheduler
                scheduler.step()


            # print every epoch
                print(f'T={T} Epoch: {epoch+1}/{NUM_EPOCHS} - Loss: {tot_loss:.4f} b_loss={best_loss:.4f} - T: {time.time()-t0:.2f}s/epoch ,tempo_previto={((time.time()-t0)*(NUM_EPOCHS-1-epoch+NUM_EPOCHS*(len(qc_array)-q_indx-1)+NUM_EPOCHS*len(qc_array)*(len(min_array)-min_indx-1)+NUM_EPOCHS*len(qc_array)*len(min_array)*(len(layer_array)-layer_indx-1)))/60:.2f} min{min_b} nl{n_layer} QC{qc}')
                #print(f'T={T} Epoch: {epoch+1}/{NUM_EPOCHS} - Loss: {loss.item():.4f} b_loss={best_loss:.4f} - T: {time.time()-t0:.2f}s/epoch ,tempo_previto={(((NUM_EPOCHS-1-epoch+NUM_EPOCHS*(len(qc_array)-q_indx-1)+NUM_EPOCHS*len(qc_array)*(len(min_array)-min_indx-1)+NUM_EPOCHS*len(qc_array)*len(min_array)*(len(layer_array)-layer_indx-1)))):.2f} min{min_b} nl{n_layer}')
                
            np.save(f'thetas/all_thetas_T{T}_nl{n_layer}_min{min_b}_qc{qc}_{Q_ANCILLA}_ld{ld_dim}_cond_all.npy',trained_thetas_1)
            np.save(f'losses/all_loss__T{T}_nl{n_layer}_min{min_b}_qc{qc}_ancilla{Q_ANCILLA}_ld{ld_dim}_cond_all.npy',loss_history)