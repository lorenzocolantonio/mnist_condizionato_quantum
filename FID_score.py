import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D



ld_dim=4
count=0
Q_ANCILLA=5
n_samples=1000
T_array=np.array([5,10])
qc_array=np.array([64,96,112,120,124])
min_array=np.array([0.05,0.01,0.005])
layer_array=np.array([5,10,20,50])
all=np.load(f'dataset_ld_{ld_dim}_{9}.npy')
all=all
for i in range(9):
    x=np.load(f'dataset_ld_{ld_dim}_{i}.npy')
    all=np.concatenate((x,all))

dataset2=all
print(np.shape(dataset2))

all=[]

def make_matrix_positive_semidefinite(cov_matrix):
    # Perform eigenvalue decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Set negative eigenvalues to zero
    eigenvalues[eigenvalues < 0] = 1e-30

    # Reconstruct the positive semidefinite matrix
    positive_semidefinite_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

    return positive_semidefinite_matrix

dataset2_mean = np.mean(dataset2, axis=0) #(16,)
dataset2_cov = np.cov(dataset2, rowvar=False)#(16,16)
dataset2_cov=make_matrix_positive_semidefinite(dataset2_cov)

for T in T_array:
    for n_layer in layer_array:
         qc_best=[]
         for qc in qc_array:
            all=[]
            for min in min_array:
                dataset1=np.squeeze(np.load(f'all_ancilla{Q_ANCILLA}/all_generated_ld{ld_dim}_T{T}_nl{n_layer}_min{min}_qc{qc}_{Q_ANCILLA}.npy')) #(1000,ld_dim)
                print(np.shape(dataset1))
                #quit()
                #print(np.shape(dataset1))
                
                
                
                
                dataset1_mean = np.mean(dataset1, axis=0) #(16,)
                
                dataset1_cov = np.cov(dataset1, rowvar=False)
                
                dataset1_cov=make_matrix_positive_semidefinite(dataset1_cov)
                
               #print(np.linalg.eigvals(dataset1_cov))
                
              
                
                
                mean_diff = dataset1_mean - dataset2_mean
                mean_diff_squared = np.dot(mean_diff, mean_diff)
                

                '''plt.imshow(np.dot(dataset1_cov, dataset2_cov),cmap='gray')
                plt.show()'''
                
                
                
                cov_sqrt = sqrtm(np.dot(dataset1_cov, dataset2_cov))
                trace = np.trace(cov_sqrt)
                

                fid = mean_diff_squared + np.trace(dataset1_cov) + np.trace(dataset2_cov) - 2 * trace
                
                
                all=np.append(all,fid)

            best=np.min(all)
            qc_best=np.append(qc_best,best)
         plt.title(f"T={T}")
         plt.xlabel(f'qc_compression')
         plt.xlabel(f'FID_score')
         plt.scatter(qc_array,qc_best,label=f'num_layers={n_layer}')
         plt.legend()
    plt.savefig(f'fid_score_T{T}_ld{ld_dim}_between_latent_ancilla_{Q_ANCILLA}.png')
    plt.close()
        