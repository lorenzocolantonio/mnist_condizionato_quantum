from functions import *
from NeuralNetwork import *
import numpy as np
from torch.utils.data import ConcatDataset

# create path "/Data/" if it does not exist
if not os.path.exists("Data/"): os.makedirs("Data/")
ld_arr=np.array([4,8,16,32])
device = which_device()
# Define the paths for the dataset files
data_path = os.path.expanduser("~/.torch/datasets/")
mnist_path = os.path.join(data_path, "MNIST")

# load the dataset
train_loader, test_loader = download_and_load_mnist(BATCH_SIZE, shuffle=True)
for l_indx in range(len(ld_arr)):
    latent_dim=ld_arr[l_indx]
    encoder = Encoder(latent_dim).to(device)

    encoder.load_state_dict(torch.load(f"Weights/encoder_ld{latent_dim}_MSE.pth", map_location=device))

    

    dataset = ConcatDataset([train_loader.dataset, test_loader.dataset])

    batched_dataset = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)

    latent_dataset = []
    latent_targets = []

    for batch_idx, (data, target) in enumerate(batched_dataset):

        latent_batch = encoder(data.to(device))
        min_val, _= torch.min(latent_batch,dim=1)
        latent_batch=latent_batch-min_val.unsqueeze(1)
        latent_batch = F.normalize(latent_batch, dim=1)
        

        # store the latent vectors
        latent_dataset.append(latent_batch.detach().cpu())
        latent_targets.append(target)

    latent_dataset = torch.cat(latent_dataset, dim=0)
    latent_targets = torch.cat(latent_targets, dim=0)

    # save the latent vectors
    torch.save(latent_dataset, f"Data/latent_dataset_ld{latent_dim}_MSE.pth")
    torch.save(latent_targets, f"Data/latent_targets_ld{latent_dim}_MSE.pth")
    print(l_indx+1)
