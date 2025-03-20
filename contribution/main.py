import torch
import matplotlib.pyplot as plt
from data import MNISTData
from vae import LipschitzVAE
from utils import plot_reconstructions
from pac_bayes import recons_bound_diam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 50
batch_size = 256
train_size = 50000
n = 20000
test_size = 20000
lamda = n / 1 # You can change the value of lambda (but you also need to change the kl_coeff)


if __name__ == '__main__':
    mnist_data = MNISTData(batch_size=batch_size)
    
    mnist_model = LipschitzVAE(
        encoder_hidden_layers=[400, 200],
        decoder_hidden_layers=[200, 400],
        input_dim=mnist_data.input_dim,
        latent_dim=20,
        bias=True,
        config_name='config.json'
    ).to(device) 
    
    mnist_model.train_model(train_loader=mnist_data.train_loader, epochs=epochs, kl_coeff=1/n, lr=1e-4)

    torch.save(mnist_model.state_dict(), "mnist_vae.pth")
    print("Model saved in mnist_vae.pth")
    
    mnist_bound_dico = recons_bound_diam(
        model=mnist_model,
        val_loader=mnist_data.val_loader,
        test_loader=mnist_data.test_loader,
        lamda=lamda,
        k_phi=2,
        k_theta=2,
        delta=0.05,
        diameter=1.0,  # MNIST normalisé entre 0 et 1
        prior=mnist_model.prior
    )

    print("\n Results:")
    for key, value in mnist_bound_dico.items():
      print(f" {key}: {value}")
    
    mnist_model.eval()
    test_loader = mnist_data.test_loader
    batch, _ = next(iter(test_loader)) 
    batch = batch.to(device)


    # Reconstruction of the images
    with torch.no_grad():
        z, _, _ = mnist_model.encode(batch.view(batch_size, -1))  
        reconstructed_images = mnist_model.decode(z).cpu()  

    plot_reconstructions(batch, reconstructed_images, n=8)

    mnist_model.history.show_losses(['rec_loss', 'kl_div', 'loss'], start_epoch=1)
    plt.savefig("training_losses.png") 
    print("figure saved in mnist_reconstructions.png")
    plt.show()
