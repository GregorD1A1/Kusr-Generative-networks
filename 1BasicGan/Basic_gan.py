import torch
from torch.utils.data import DataLoader
from torch import nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# visualization function
def show(tensor, n_channels=1, size=(28, 28), n_images=16):
    print(tensor.shape)
    data = tensor.detach().cpu().view(-1, n_channels, *size)
    print(data.shape)
    grid = make_grid(data[:n_images], nrow=4).permute(1, 2, 0)
    plt.imshow(grid)
    plt.show()


# generator
def gen_block(input, out):
    return nn.Sequential(
        nn.Linear(input, out),
        nn.BatchNorm1d(out),
        nn.ReLU(inplace=True),
    )


class Generator(nn.Module):
    def __init__(self, z_dim=64, i_dim=28*28, h_dim=128):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            gen_block(z_dim, h_dim),     # 64, 128
            gen_block(h_dim, h_dim*2),   # 128, 256
            gen_block(h_dim*2, h_dim*4), # 256, 512
            gen_block(h_dim*4, h_dim*8), # 512, 1024
            nn.Linear(h_dim*8, i_dim),   # 1024, 784
            nn.Sigmoid(),
        )

    def forward(self, noice_vektor):
        return self.gen(noice_vektor)


def gen_noice_vektor(n_vectors, z_dim):
    return torch.randn(n_vectors, z_dim).to(device)


# discriminator
def discriminator_block(input, out):
    return nn.Sequential(
        nn.Linear(input, out),
        nn.LeakyReLU(0.2),
    )


class Discriminator(nn.Module):
    def __init__(self, i_dim=28*28, h_dim=256):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            discriminator_block(i_dim, h_dim*4),
            discriminator_block(h_dim*4, h_dim*2),
            discriminator_block(h_dim*2, h_dim),
            nn.Linear(h_dim, 1),
        )

    def forward(self, image):
        return self.disc(image)

# hyperparameters
epochs = 200
z_dim = 64  # wymiary wektora cech
lr = 0.00001
batch_size = 128


# parameters
mean_gen_loss = 0
mean_disc_loss = 0
mean_fakes_as_trues = 0
number_to_draw = 6

loss_fcn = nn.BCEWithLogitsLoss()    # binary cross entropy z sigmoidem przed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = MNIST('data', download=True, transform=transforms.ToTensor())
# wyciągamy określoną liczbę z datasetu
dataset = [dataset[x] for x in range(len(dataset)) if dataset[x][1] == number_to_draw]

dataloader = DataLoader(dataset, shuffle=True,
                        batch_size=batch_size)

generator = Generator(z_dim=z_dim).to(device)
gen_optim = torch.optim.Adam(generator.parameters(), lr=lr)
discriminator = Discriminator().to(device)
disc_optim = torch.optim.Adam(discriminator.parameters(), lr=lr)


for epoch in range(epochs):
    for cur_step, (real_x_batch, _) in enumerate(tqdm(dataloader)):

        cur_batch_size = len(real_x_batch)
        # przeprocessing realnych danych
        real_x_batch = real_x_batch.view(cur_batch_size, -1)    # wypłaszczamy obrazy
        real_x_batch = real_x_batch.to(device)


        # diskriminator
        disc_optim.zero_grad()
        # generacja fejkowych obrazów
        noise_vector_batch = gen_noice_vektor(cur_batch_size, z_dim)
        fake_x_batch = generator(noise_vector_batch)
        # forward pass
        fake_pred_y_batch = discriminator(fake_x_batch.detach())    # detaczujemy by nie zepsuć parametrów tensora
        real_pred_y_batch = discriminator(real_x_batch.detach())    # pod liczenie gradientów dla generatora

        fake_targets = torch.zeros_like(fake_pred_y_batch)
        real_targets = torch.ones_like(real_pred_y_batch)
        # loss and optimizer
        loss_for_fakes = loss_fcn(fake_pred_y_batch, fake_targets)
        loss_for_reals = loss_fcn(real_pred_y_batch, real_targets)
        disc_loss = (loss_for_reals + loss_for_fakes) / 2
        disc_loss.backward(retain_graph=True)   # zachowujemy graf obliczeń pochodnych na przyszłość
        disc_optim.step()

        # do celów statystycznych
        sigmoid = nn.Sigmoid()
        n_fakes_as_trues = torch.round(sigmoid(fake_pred_y_batch)).sum().item()
        mean_fakes_as_trues += n_fakes_as_trues


        # generator
        gen_optim.zero_grad()
        # generacja fejkowych obrazów
        noise_vector_batch = gen_noice_vektor(cur_batch_size, z_dim)
        fake_x_batch = generator(noise_vector_batch)
        # forward pass
        pred_y_batch = discriminator(fake_x_batch)
        targets = torch.ones_like(pred_y_batch)
        # loss and optimizer
        gen_loss = loss_fcn(pred_y_batch, targets)
        gen_loss.backward(retain_graph=True)
        gen_optim.step()


        # obliczenie średniej straty do wizualizacji
        mean_disc_loss += disc_loss.item()
        mean_gen_loss += gen_loss.item()

    # wyświetlanie rysunków i średniej po każdej kolejnej epoce
    n_steps = cur_step + 1
    mean_disc_loss = mean_disc_loss / n_steps
    mean_gen_loss = mean_gen_loss / n_steps
    mean_fakes_as_trues = round(mean_fakes_as_trues / n_steps, 1)
    show(fake_x_batch)
    show(real_x_batch)
    print(f'epoka {epoch}/{epochs}; średnia strata generatora:{mean_gen_loss},',
          f'średnia strata diskriminatora: {mean_disc_loss},'
          f'srednia ilosc oszukanych: {mean_fakes_as_trues}/{batch_size}')
    mean_gen_loss, mean_disc_loss = 0, 0
    mean_fakes_as_trues = 0
