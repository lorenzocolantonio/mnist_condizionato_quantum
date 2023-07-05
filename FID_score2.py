import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
#qc_array=np.array([32,48,56,60,62])
#qc_array=np.array([64,96,112,120,124])
ld_dim=4
count=0
Q_ANCILLA=0
n_samples=1000
T_array=np.array([5,10])
qc_array=np.array([0,2,3])
min_array=np.array([0.05,0.01,0.005])
layer_array=np.array([5,10,20,50])



dataset2=np.load(f'true_images_mnist.npy')
dataset2=np.reshape(dataset2,(-1,28*28))
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
                count+=1
                print(count)
                dataset1=np.squeeze(np.load(f'all_ancilla{Q_ANCILLA}/img_compressed_ld{ld_dim}_T{T}_nl{n_layer}_qc{qc}_min{min}_qa_{Q_ANCILLA}.npy')) #(1000,ld_dim)
                print(np.shape(dataset1))
                
                
                
                
                
                
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
         plt.title(f"T={T} FID_score generated images vs true images")
         plt.xlabel(f'qc_compression')
         plt.xlabel(f'FID_score')
         plt.scatter(qc_array,qc_best,label=f'num_layers={n_layer}')
         plt.legend()
    plt.savefig(f'fid_score_T{T}_ld{ld_dim}_vstrueimg_{Q_ANCILLA}.png')
    plt.close()