import numpy as np
import torch
import matplotlib.pyplot as plt

from Modules.pennylane_functions import *
from Modules.training_functions import *
classes=10
n_samples=100
T_array=np.array([5])
qc_array=np.array([120,124,126])
min_array=np.array([0.1,0.2])
beta_array=np.array([10e-5,10e-3,10e-2])
layer_array=np.array([50,75]) 
count=0
zero = torch.zeros((n_samples, 2**NUM_QUBITS), dtype=torch.complex64)
for T in T_array:
     for n_layer in layer_array:
         for qc in qc_array:
            for min in min_array:
                for beta in beta_array:
                    all_thetas=np.load(f'thetas/all_thetas_T{T}_nl{n_layer}_min{min}_beta{beta}_qc{qc}_{Q_ANCILLA}_ld{ld_dim}_cond_all.npy')
                    
                    thetas=all_thetas[12]
                    print(np.shape(thetas))
                    thetas=torch.tensor(thetas)
                    
                    for j in range(classes):
                        
        
                
                        noise_batch = torch.randn(n_samples, ld_dim,2)
                        noise_batch=torch.view_as_complex(noise_batch)
                        zero_noise=zero
                        zero_noise[:,j*ld_dim:(j+1)*ld_dim]=noise_batch
                        
                        denoised_batch = zero_noise
                        to_save=[]
                        history_dn=[]

                        
                        # implement denoising loop

                        for i in range(T):
                                denoised_batch=denoised_batch/torch.norm(denoised_batch, dim = 1).view(-1, 1)
                                denoised_batch = circuit_aq(qc,j,thetas,n_layer, denoised_batch)
                                temp=denoised_batch
                                denoised_batch=zero
                                
                                denoised_batch[:,j*ld_dim:(j+1)*ld_dim]=temp[:,j*ld_dim:(j+1)*ld_dim]
                                history_dn.append(torch.abs(denoised_batch[:,j*ld_dim:(j+1)*ld_dim]).detach().numpy())
                                
                        to_save.append(torch.abs(denoised_batch[:,j*ld_dim:(j+1)*ld_dim]).detach().numpy())
                        #print(np.shape(to_save))
                        count+=1
                        print(count/(len(T_array)*len(qc_array)*len(layer_array)*len(min_array)))
                        np.save(f'generated_cond/all_all_generated_ld{ld_dim}_T{T}_nl{n_layer}_min{min}_beta{beta}_qc{qc}_{Q_ANCILLA}_{j}_prova.npy',to_save)

                        np.save(f'generated_cond/all_history_generated_ld{ld_dim}_T{T}_nl{n_layer}_min{min}_beta{beta}_qc{qc}_{Q_ANCILLA}_{j}_prova.npy',history_dn)

                