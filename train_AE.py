from torch import optim
import matplotlib.pyplot as plt
from functions import *
from NeuralNetwork import *
import time
import numpy as np
device = which_device()

if not os.path.exists("Weights/"): os.makedirs("Weights/")
#ld_array=np.array([4])
ld_array=np.array([4,8,16,32])
train_loader, test_loader = download_and_load_mnist(BATCH_SIZE, shuffle=True)
for latent_indx in range(len(ld_array)):
    lat_dim=ld_array[latent_indx]

    # define autoencoder
    encoder = Encoder(lat_dim).to(device)
    decoder = Decoder(lat_dim).to(device)

    # define loss function and optimizer
    critereon = nn.L1Loss()
    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor = 0.5 , verbose=True)

    best_loss = 1e9
    losses=[]
    # train autoencoder
    for epoch in range(1, NUM_EPOCHS + 1):
        t0=time.time()

        train_loss = train(encoder, decoder, train_loader, critereon, optimizer, device)
        test_loss = eval(encoder, decoder, test_loader, critereon, device)
        losses.append(train_loss)
        # save best model
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(encoder.state_dict(), f"Weights/encoder_ld{lat_dim}_MAE.pth")
            torch.save(decoder.state_dict(), f"Weights/decoder_ld{lat_dim}_MAE.pth")
            print("Best model saved!")
        print(f"lat_dim={lat_dim}Epoch: {epoch}/{NUM_EPOCHS}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, best_loss={best_loss:.2f} s/epoch={time.time()-t0:.4f} tempo previsto={((time.time()-t0)*(NUM_EPOCHS-epoch))/60:.4f}min")
        plt.plot(losses)
        plt.savefig(f'loss_MAE_ld{lat_dim}')
        plt.close()
        scheduler.step(test_loss)

    