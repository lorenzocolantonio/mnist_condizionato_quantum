from functions import *
from NeuralNetwork import *
import numpy as np  
import matplotlib.pyplot as plt
from functions import *
from NeuralNetwork import *
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = which_device()
digit=4
ld_dim=16
count=0
Q_ANCILLA=4
n_samples=100
T_array=np.array([5])

qc_array=np.array([248,252,254])
min_array=np.array([0.1,0.2])
layer_array=np.array([20,50]) 
encoder = Encoder(ld_dim).to(device)
decoder = Decoder(ld_dim).to(device)
encoder.load_state_dict(torch.load(f"Weights/encoder_ld{ld_dim}_MAE.pth", map_location=device))
decoder.load_state_dict(torch.load(f"Weights/decoder_ld{ld_dim}_MAE.pth", map_location=device))
for T in T_array:
    for n_layer in layer_array:
         for qc in qc_array:
            for min in min_array:
                
                #gen_quantum=np.load(f'generated_cond/all_all_generated_ld{ld_dim}_T{T}_nl{n_layer}_min{min}_qc{qc}_{Q_ANCILLA}_{digit}_metodo2_1.npy')
                gen_denoise=np.load(f'generated_cond/all_history_generated_ld{ld_dim}_T{T}_nl{n_layer}_min{min}_qc{qc}_{Q_ANCILLA}_{digit}_prova.npy')
                print(np.shape(gen_denoise))
                #gen_denoise=np.random.normal(size=(5,100,8))
                
                print(np.shape(gen_denoise[0]))
                fig, axes = plt.subplots(5, 5, figsize=(8, 8))
                for t in range(T):
                    

                    gen_quantum_1=torch.tensor(gen_denoise[t])
                    
                    
                    latent=gen_quantum_1.to(device)
                    latent=latent.to(torch.float32)
                    
                    outputs = decoder(latent).cpu().detach().numpy()
                    outputs_1=np.reshape(outputs,(n_samples,-1))
                    print(np.shape(outputs_1))
                    
                    
        

                    # Itera attraverso le immagini e i subplot corrispondenti
                    for i in range(5):
                        # Ottieni l'immagine corrente
                        axes[i,t].imshow(outputs_1[i].reshape(28,28))
                        
                        
                        
                plt.show()

                    