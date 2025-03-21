import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np

from torch.distributions import Normal 

from lnets.models.activations import GroupSort
from lnets.models.architectures.base_architecture import Architecture
from lnets.models.layers.dense.bjorck_linear import BjorckLinear
from lnets.models.layers.scale import Scale

from loss import Loss
from history import History
from utils import get_device, get_config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseVAE(nn.Module):
    def __init__(self, encoder=None, decoder=None, loss=None):
        super(BaseVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

        self.loss_fct = loss if loss is not None else Loss(criterion='mse')
        self.history = History()

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def training_step(self, x):
        raise NotImplementedError

    def train_model(self, train_loader, epochs, kl_coeff, lr):
        optimizer = Adam(params=self.parameters(), lr=lr)
        for epoch in range(1, epochs + 1):
            print('Epoch {} ...'.format(epoch))
            rec_losses, kl_divs, losses = [], [], []
            for batch, _ in tqdm(train_loader):
                batch = batch.to(device)
                rec_loss, kl_div = self.training_step(batch)
                loss = rec_loss + kl_coeff * kl_div

                # backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                rec_losses.append(float(rec_loss.detach()))
                kl_divs.append(float(kl_div.detach()))
                losses.append(float(loss.detach()))

            # End of epoch
            avg_rec, avg_kl, avg_loss = np.mean(rec_losses), np.mean(kl_divs), np.mean(losses)
            print('rec_loss: {};   kl_div: {};   loss: {}'.format(avg_rec, avg_kl, avg_loss))
            logs = {'rec_loss': avg_rec, 'kl_div': avg_kl, 'loss': avg_loss}
            self.history.save(logs)

        print('End of training!')

    def generate(self, num_samples):
        z = self.prior.sample(num_samples).to(device) 
        with torch.no_grad():
            samples = self.decoder(z).to(device)
        return samples


class GaussianMixturePrior:
    """Implements a Gaussian Mixture (GMM) based prior for the latent space of a VAE."""
    def __init__(self, num_components, latent_dim):
        self.num_components = num_components
        self.latent_dim = latent_dim
        self.mus = torch.randn(num_components, latent_dim)  
        self.sigmas = torch.ones(num_components, latent_dim)  
        self.weights = torch.ones(num_components) / num_components  

    def to(self, device):
        self.mus = self.mus.to(device)
        self.sigmas = self.sigmas.to(device)
        self.weights = self.weights.to(device)
        return self  

    def sample(self, batch_size):
        k = torch.multinomial(self.weights, batch_size, replacement=True)
        mus = self.mus[k]
        sigmas = self.sigmas[k]
        return torch.normal(mus, sigmas)

    def log_prob(self, z):
      z = z.to(self.mus.device)  
      probs = []
      for k in range(self.num_components):
          normal = torch.distributions.Normal(self.mus[k].to(z.device), self.sigmas[k].to(z.device))
          probs.append(normal.log_prob(z).sum(dim=-1) + torch.log(self.weights[k].to(z.device)))
      return torch.logsumexp(torch.stack(probs), dim=0)  


class LipschitzVAE(Architecture, BaseVAE):
    def __init__(self, encoder_hidden_layers, decoder_hidden_layers, input_dim, latent_dim, bias=True, config_name=None):
        super(LipschitzVAE, self).__init__()
        self.config = get_config(config_name)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_bias = bias

        self.encoder_layer_sizes = [input_dim] + encoder_hidden_layers + [2*latent_dim]
        self.decoder_layer_sizes = [latent_dim] + decoder_hidden_layers + [input_dim]

        self.k_phi = self.config.model.encoder.l_constant
        self.k_theta = self.config.model.decoder.l_constant

        self.encoder_num_layers = len(self.encoder_layer_sizes)
        self.decoder_num_layers = len(self.decoder_layer_sizes)

        self.encoder_groupings = [-1] + self.config.model.encoder.groupings
        self.decoder_groupings = [-1] + self.config.model.decoder.groupings

        encoder_layers = self._get_layers(layer_sizes=self.encoder_layer_sizes,
                                          groupings=self.encoder_groupings,
                                          l_constant_per_layer=self.k_phi ** (
                                                                      1.0 / (self.encoder_num_layers - 1)),
                                          config=self.config)
        self.encoder = nn.Sequential(*encoder_layers).to(device)
        self.sigmoid = nn.Sigmoid()

        decoder_layers = self._get_layers(layer_sizes=self.decoder_layer_sizes,
                                          groupings=self.decoder_groupings,
                                          l_constant_per_layer=self.k_theta ** (
                                                                 1.0 / (self.decoder_num_layers - 1)),
                                          config=self.config)
        self.decoder = nn.Sequential(*decoder_layers).to(device)
        self.prior = GaussianMixturePrior(num_components=5, latent_dim=self.latent_dim).to(device)


    def _get_layers(self, layer_sizes, groupings, l_constant_per_layer, config):
        layers = list()
        layers.append(BjorckLinear(layer_sizes[0], layer_sizes[1], bias=self.use_bias, config=config))
        layers.append(Scale(l_constant_per_layer, cuda=self.config.cuda))

        for i in range(1, len(layer_sizes) - 1):
            # Add the activation layer
            layers.append(GroupSort(layer_sizes[i] // groupings[i]))

            # Add the linear layer + scaling
            layers.append(BjorckLinear(int(layer_sizes[i]), layer_sizes[i + 1], bias=self.use_bias, config=config))
            layers.append(Scale(l_constant_per_layer, cuda=self.config.cuda))
        return layers

    def update_gmm_centers(self, train_loader):
        """
        Updates the GMM prior centers with the true encoded latents.
        """
        with torch.no_grad():
            latents = []
            for x, _ in train_loader:
                z, _, _ = self.encode(x.to(device)) 
                latents.append(z.cpu().numpy())

            latents = np.concatenate(latents, axis=0)
            new_mus = np.mean(latents.reshape(self.prior.num_components, -1, self.latent_dim), axis=1)
            self.prior.mus = torch.tensor(new_mus, dtype=torch.float32, device=device)


    def encode(self, x):
        x = x.view(-1, self.input_dim).to(device)
        encoder_out = self.encoder(x)
        mu = encoder_out[:, :self.latent_dim]
        std = self.sigmoid(encoder_out[:, self.latent_dim:])

        z = mu + std * self.prior.sample(mu.shape[0]).to(device) #TIF
        return z, mu, std

    def decode(self, z, shape=None):
        z = z.to(device) 
        recons = self.decoder(z)
        recons = recons.view(shape) if shape is not None else recons
        return recons

    def training_step(self, x):
        z, mu, std = self.encode(x)
        x_hat = self.decode(z)
        rec_loss, kl_div = self.loss_fct(x=x, x_hat=x_hat, mu=mu, logvar=torch.log(torch.pow(std, 2)), prior=self.prior)
        return rec_loss, kl_div
