import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, nb_units=512):
        super(Generator, self).__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        modules = nn.Sequential(
            nn.Linear(self.input_dim, nb_units),
            nn.BatchNorm1d(nb_units, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(nb_units, self.output_dim),
            nn.Tanh()
        )
        self.model = modules
    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim, nb_units=512):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim
        modules = nn.Sequential(
            nn.Linear(self.input_dim, nb_units),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(nb_units, 1),
            nn.Sigmoid()
        )
        self.model = modules
    def forward(self, img):
        return self.model(img)

