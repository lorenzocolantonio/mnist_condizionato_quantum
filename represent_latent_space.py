import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
ld_arr=np.array([4,8,16,32])
for l_indx in range(len(ld_arr)):
    latent_dim=ld_arr[l_indx]
    # load Data/latent_dataset.pth and Data/latent_targets.pth
    latent_dataset = torch.load(f"Data/latent_dataset_ld{latent_dim}_MSE.pth").numpy()
    latent_targets = torch.load(f"Data/latent_targets_ld{latent_dim}_MSE.pth").numpy()
    ld_arr=np.array([4,8,16,32])
    tsne = TSNE(n_components=2, random_state=42, verbose=1)

    latent_2d = tsne.fit_transform(latent_dataset)

    # save latent_2d.pth
    torch.save(latent_2d, f"Data/latent_dataset_compressed_ld{latent_dim}_MSE.pth")

    # load latent_2d.pth
    latent_2d = torch.load(f"Data/latent_dataset_compressed_ld{latent_dim}_MSE.pth")

    plt.figure(figsize=(10, 8))

    # Loop through the unique labels (1 to 9)
    for label in range(0, 10):
        indices = np.where(latent_targets == label)
        plt.scatter(latent_2d[indices, 0], latent_2d[indices, 1], label=f"Label {label}", s = 2.5)

    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend(markerscale=6)
    plt.title("Visualization of Latent Dataset")
    plt.savefig(f'tsne_ld{latent_dim}_MSE')
    plt.close()
