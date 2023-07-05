import torch
import numpy as np
import matplotlib.pyplot as plt

from Modules.pennylane_functions import *
from Modules.hyperparameters import *

# perfrom t noise steps (formula (4) in the DDPM paper)
def assemble_input(image_batch, t, alphas_bar,lat_dim, device):

    # compute batch of coefficients
    alphas_bar_batch = alphas_bar[t]

    # compute averages
    avgs = torch.sqrt(alphas_bar_batch).unsqueeze(1).repeat(1, lat_dim)*image_batch
    
    noise= torch.randn((BATCH_SIZE,lat_dim,2),device=device)
    complex_noise = torch.view_as_complex(noise)

    # compute fluctuations
    fluct = (1-alphas_bar_batch).unsqueeze(1).repeat(1, lat_dim)*complex_noise
    # perform t noise steps
    noisy_image = avgs+fluct

    return noisy_image






# perfrom a noise step (formula (2) in the DDPM paper)
def noise_step(image_batch, t, betas,lat_dim, device):

        # compute batch of coefficients
        betas_batch = betas[t]

        # compute averages by rescaling the input image
        avgs = torch.sqrt(1-betas_batch).unsqueeze(1).repeat(1, lat_dim)*image_batch

        # get fluctuations by rescaling Gaussian noise
        noise= torch.randn((BATCH_SIZE,lat_dim,2),device=device)
        complex_noise = torch.view_as_complex(noise)
        
        #fluct = betas[t].unsqueeze(1).repeat(1, 256)*torch.randn(image_batch.shape, device=device)
        fluct = betas[t].unsqueeze(1).repeat(1,lat_dim)*complex_noise
    
        # perform noise step
        noisy_image_batch = avgs+fluct

       # rescale each image in the batch between -1 and 1
        #min_img = torch.min(noisy_image_batch, dim = 1)[0].view(-1, 1)
        #max_img = torch.max(noisy_image_batch, dim = 1)[0].view(-1, 1)
        #noisier_image_batch = (noisy_image_batch - min_img) / (max_img - min_img) * 2 - 1

        return noisy_image_batch 


def loss_fn_aq(qc,weights,n_layer ,input_batch, target_batch,label):
    # Compute the output states
    output_batch_temp = circuit_aq(qc,label,weights,n_layer, input_batch)
    output_batch=torch.zeros((len(output_batch_temp), 2**NUM_QUBITS), dtype=torch.complex64)
    
    output_batch[:,label*ld_dim:(label+1)*ld_dim]=output_batch_temp[:,label*ld_dim:(label+1)*ld_dim]
    #print(output_batch)
    #print(torch.norm(output_batch, dim = 1).view(-1, 1))

    
    output_batch= output_batch/torch.norm(output_batch, dim = 1).view(-1, 1)
    target_batch = target_batch/torch.norm(target_batch, dim = 1).view(-1, 1)
    #if label==8 :print(output_batch)
    '''print(output_batch[:,:ld_dim])
    print(target_batch[:,:ld_dim])
    print(torch.sum(torch.conj(output_batch)*target_batch, dim=1))'''

    # compute the fidelity between the output states and the target states
    fid = torch.abs(torch.sum(torch.conj(output_batch)*target_batch, dim=1))**2
    fid=fid.mean()
    if torch.isnan(fid):
         fid=0
    #print(np.shape(fid))
    

    # Return the mean of the fidelity for the batch
    return 1-fid
# Define the cost function

