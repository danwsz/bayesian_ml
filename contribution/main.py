import matplotlib.pyplot as plt
import torch

from data import MNISTData
from vae import LipschitzVAE
from utils import heat_scatter, plot_generated_images
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
    print("Modèle sauvegardé dans mnist_vae.pth")
    
    # Borne PAC-Bayésienne (en ajustant les paramètres selon tes besoins)
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

    print("PAC-Bayes bound for MNIST:", mnist_bound_dico)

    
    generated_samples = mnist_model.generate(64).to(device) 
    plot_generated_images(generated_samples, image_shape=(28, 28), n_row=8)  # Suppression du unpacking
    plt.savefig("generated_mnist_images_loaded.png")  # Sauvegarde de l'image générée
    plt.show()
