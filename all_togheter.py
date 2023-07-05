import numpy as np
ld_dim=4
all=np.load(f'dataset_ld_{ld_dim}_{9}.npy')
all=all[:200]
for i in range(9):
    x=np.load(f'dataset_ld_{ld_dim}_{i}.npy')
    all=np.concatenate((x[:200],all))
print(np.shape(all))