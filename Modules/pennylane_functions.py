import torch
import pennylane as qml
import numpy as np
from Modules.hyperparameters import *

# get the device
Ldev = qml.device("default.qubit", wires=NUM_QUBITS)


Ldev = qml.device("default.qubit", wires=NUM_QUBITS)
@qml.qnode(Ldev)
def Lblock(thetas, state=None):

    # Load the initial state if provided
    if state is not None: qml.QubitStateVector(state, wires=range(NUM_QUBITS))

    # create the parameterized circuit
    qml.StronglyEntanglingLayers(thetas.reshape(1, NUM_QUBITS, 3), wires=range(NUM_QUBITS))

    # Return the state vector
    return qml.state()

E_dev = qml.device("default.qubit", wires=NUM_QUBITS)
@qml.qnode(E_dev)
def encoder_block(n_layer_e,thetas, state=None):

    # Load the initial state if provided
    if state is not None: qml.QubitStateVector(state, wires=range(NUM_QUBITS))

    # create the parameterized circuit
    qml.StronglyEntanglingLayers(thetas.reshape(n_layer_e, NUM_QUBITS, 3), wires=range(NUM_QUBITS))

    # Return the state vector
    return qml.state()

D_dev = qml.device("default.qubit", wires=NUM_QUBITS)
@qml.qnode(D_dev)
def decoder_block(n_layer_d,thetas, state=None):

    # Load the initial state if provided
    if state is not None: qml.QubitStateVector(state, wires=range(NUM_QUBITS))

    # create the parameterized circuit
    qml.StronglyEntanglingLayers(thetas.reshape(n_layer_d, NUM_QUBITS, 3), wires=range(NUM_QUBITS))

    # Return the state vector
    return qml.state()

latent_dev = qml.device("default.qubit", wires=NUM_QUBITS)
@qml.qnode(latent_dev, interface="torch")
def latent_block(weights, state=None):
    # Load the initial state if provided
    if state is not None: qml.QubitStateVector(state, wires=range(NUM_QUBITS))
    
    # create the parameterized circuit
    qml.RX(0, wires=0)
    qml.RY(0, wires=0)
    qml.RZ(0, wires=0)
    for q in range(NUM_QUBITS-1):
        qml.RX(weights[3*q], wires=q+1)
        qml.RY(weights[3*q + 1], wires=q+1)
        qml.RZ(weights[3*q + 2], wires=q+1)
    qml.Barrier()

    '''# create the entangling layer
    for n in range(NUM_QUBITS-2):
        qml.CNOT(wires=[n+1, (n+2)%NUM_QUBITS])
'''
    # Return the state vector
    return qml.state()




# get the device
noise_dev = qml.device("default.qubit", wires=NUM_QUBITS)
@qml.qnode(noise_dev, interface="torch")
def noise_block(weights, state=None):
    # Load the initial state if provided
    
    if state is not None: qml.QubitStateVector(state, wires=range(NUM_QUBITS))
    # create the parameterized circuit
    
    for q in range(NUM_QUBITS):
        qml.RX(weights[3*q], wires=q)
        qml.RY(weights[3*q + 1], wires=q)
        qml.RZ(weights[3*q + 2], wires=q)
        
    # create the entangling layer
    
    for n in range(NUM_QUBITS):
        qml.CNOT(wires=[n, (n+1)%(NUM_QUBITS)])
        
    


    for q in range(NUM_QUBITS):
        qml.RX(weights[3*q+NUM_QUBITS*3], wires=q)
        qml.RY(weights[3*q + 1+NUM_QUBITS*3], wires=q)
        qml.RZ(weights[3*q + 2+NUM_QUBITS*3], wires=q)
    

    for n in range(NUM_QUBITS):
         qml.CNOT(wires=[(NUM_QUBITS-n-1), (NUM_QUBITS- n)%(NUM_QUBITS)])
         
    for q in range(NUM_QUBITS):
        qml.RX(weights[3*q+NUM_QUBITS*6], wires=q)
        qml.RY(weights[3*q + 1+NUM_QUBITS*6], wires=q)
        qml.RZ(weights[3*q + 2+NUM_QUBITS*6], wires=q)
        
    # Return the state vector
    return qml.state()






Random_dev = qml.device("default.qubit", wires=NUM_QUBITS)
# Define the linear block to be used for non-linear layers
@qml.qnode(Random_dev, interface="torch")
def Random_block(weights, state=None):

    # Load the initial state if provided
    if state is not None: qml.QubitStateVector(state, wires=range(NUM_QUBITS))
    qml.RX(0, wires=0)
    qml.RY(0, wires=0)
    qml.RZ(0, wires=0)
    for q in range(NUM_QUBITS-1):
        qml.RX(weights[3*q], wires=(q+1))
        qml.RY(weights[3*q+ 1], wires=(q+1))
        qml.RZ(weights[3*q + 2], wires=(q+1))
    
    qml.Barrier()

    # Return the state vector
    return qml.state()



Random_dev_fq = qml.device("default.qubit", wires=NUM_QUBITS)
# Define the linear block to be used for non-linear layers
@qml.qnode(Random_dev_fq, interface="torch")
def Random_block_fq(std, state=None):

    # Load the initial state if provided
    if state is not None: qml.QubitStateVector(state, wires=range(NUM_QUBITS))
    qml.RX(std*torch.randn(BATCH_SIZE), wires=0)
    qml.RY(std*torch.randn(BATCH_SIZE), wires=0)
    qml.RZ(std*torch.randn(BATCH_SIZE), wires=0)
    for q in range(NUM_QUBITS-1):
        qml.RX(0, wires=(q+1))
        qml.RY(0, wires=(q+1))
        qml.RZ(0, wires=(q+1))
    
    qml.Barrier()

    # Return the state vector
    return qml.state()







def circuit_aq(qc,label,weights,n_layer, state = None):

    # reshape the weights
    encoder_layer_weights = weights[:n_layer*3*NUM_QUBITS].reshape((n_layer, 3*NUM_QUBITS))
    decoder_layer_weights = weights[n_layer*3*NUM_QUBITS:].reshape((n_layer, 3*(NUM_QUBITS)))
    
    # build the circuit according to the layers type
    E_index = 0
    D_index = 0

    '''for layer in range(n_layer):
            state = Lblock(encoder_layer_weights[E_index], state)
            state = state/torch.norm(state, dim = 1).view(-1, 1)
            

            E_index+=1'''
    state=encoder_block(n_layer,encoder_layer_weights,state)
    state_temp = state/torch.norm(state, dim = 1).view(-1, 1)
    '''state=torch.zeros((len(state), 2**NUM_QUBITS),dtype=torch.complex64)
    state[:,label*ld_dim:(label+1)*ld_dim]=state_temp[:,label*ld_dim:(label+1)*ld_dim]
    state=state_temp'''

    state[:,:qc]=0
    state = state/torch.norm(state, dim = 1).view(-1, 1)
    
    
    '''for layer in range(n_layer):
            
            # apply unitary block
            state = Lblock(decoder_layer_weights[D_index], state)
            state = state/torch.norm(state, dim = 1).view(-1, 1)

            D_index+=1'''

    state=decoder_block(n_layer,decoder_layer_weights,state)
    state = state/torch.norm(state, dim = 1).view(-1, 1)

    return state






