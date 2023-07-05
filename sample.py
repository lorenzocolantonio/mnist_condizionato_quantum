from functions import *
from NeuralNetwork import *
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = which_device()
lat_arr=np.array([4,8,16,32])
all=[]



data_1,_,data,_=download_and_load_mnist_sample(labels=[9])
    
data=torch.cat((data, data_1), dim=0)
all=data
for digit in range(9):
        
    data_1,_,data,_=download_and_load_mnist_sample(labels=[digit])
    
    data=torch.cat((data, data_1), dim=0)
    #data=data.to(device)
    print(np.shape(data))
    all=np.append(all,data,axis=0)


print(np.shape(all))

np.save('true_images_mnist.npy',all)
quit()
'''for l_indx in range(len(lat_arr)):
        latent_dim=lat_arr[l_indx]
        encoder = Encoder(latent_dim).to(device)
        decoder = Decoder(latent_dim).to(device)



        #print(np.shape(data))
        encoder.load_state_dict(torch.load(f"Weights/encoder_ld{latent_dim}_MAE.pth", map_location=device))
        decoder.load_state_dict(torch.load(f"Weights/decoder_ld{latent_dim}_MAE.pth", map_location=device))
        latent = encoder(data)
        #### #### #### #### #### ####
        #latent = torch.abs(latent) # qua forse non va fatto il torch abs ma semplicemente sottratto il valore minimo
        min_val, _= torch.min(latent,dim=1)
        latent=latent-min_val.unsqueeze(1)
        latent=F.normalize(latent, dim=1)
        #np.save(f'dataset_ld_{latent_dim}_{digit}',latent.detach().cpu())'''
        
    #### #### #### #### #### ####
'''outputs = decoder(latent)
    print(np.shape(outputs))
    for i in range(5):
        img = outputs[i].detach().cpu().numpy().reshape(28, 28)
    
        axs[i][1].imshow(img, cmap='gray')
    plt.savefig(f'comparison_ld{latent_dim}_MSE')
    plt.close()'''

