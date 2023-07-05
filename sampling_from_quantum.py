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
print(int(2.5))
quit()
classes=10
ld_dim=8
count=0
Q_ANCILLA=4
n_samples=100
T_array=np.array([5])
qc_array=np.array([120,124,126])
min_array=np.array([0.1,0.2])
beta_array=np.array([10e-5,10e-3,10e-2])
layer_array=np.array([50,75]) 
encoder = Encoder(ld_dim).to(device)
decoder = Decoder(ld_dim).to(device)
encoder.load_state_dict(torch.load(f"Weights/encoder_ld{ld_dim}_MAE.pth", map_location=device))
decoder.load_state_dict(torch.load(f"Weights/decoder_ld{ld_dim}_MAE.pth", map_location=device))
for T in T_array:
    for n_layer in layer_array:
         for qc in qc_array:
            for min in min_array:
                for beta in beta_array:
                    for j in range(classes):
                        gen_quantum=np.load(f'generated_cond/all_all_generated_ld{ld_dim}_T{T}_nl{n_layer}_min{min}_beta{beta}_qc{qc}_{Q_ANCILLA}_{j}_prova.npy')
                        #gen_denoise=np.load(f'generated_cond/all_history_generated_ld{ld_dim}_T{T}_nl{n_layer}_min{min}_qc{qc}_{Q_ANCILLA}_{j}_prova.npy')
                        #print(np.shape(gen_denoise[0]))
                        
                        gen_quantum=torch.tensor(gen_quantum.reshape(n_samples,ld_dim))
                        print(np.shape(gen_quantum))
                        
                        latent=gen_quantum.to(device)
                        latent=latent.to(torch.float32)
                        
                        outputs = decoder(latent).cpu().detach().numpy()
                        outputs_1=np.reshape(outputs,(n_samples,-1))
                        print(np.shape(outputs_1))
                        
                        np.save(f'generated_cond/img_compressed_ld{ld_dim}_T{T}_nl{n_layer}_qc{qc}_min{min}_qa_{Q_ANCILLA}_{j}.npy',outputs_1)
                        # Prendi le prime sedici immagini
                        first_sixteen_images = outputs[:16, 0, :, :]

                        # Crea una griglia 4x4 per il plot
                        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
                        plt.title(f'generated_ld{ld_dim}_T{T}_nl{n_layer}_min{min}_qc{qc}_{Q_ANCILLA}')

                        # Itera attraverso le immagini e i subplot corrispondenti
                        for i, ax in enumerate(axes.flat):
                            # Ottieni l'immagine corrente
                            
                            image = first_sixteen_images[i]
                            
                            # Plotta l'immagine sul subplot corrente
                            ax.imshow(image, cmap='gray')
                            ax.axis('off')

                        # Mostra il plot
                        plt.savefig(f'generated_cond/immagini/metodo1.3.2/images_{ld_dim}_T{T}_nl{n_layer}_beta{beta}_min{min}_qc{qc}_{Q_ANCILLA}_{j}_prova.png')
                        plt.close(fig)
                        count+=1
                        print(count/(len(T_array)*len(qc_array)*len(layer_array)*len(min_array)))
                