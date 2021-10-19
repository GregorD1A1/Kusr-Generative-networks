import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Resize
import PIL
from PIL import Image
import os
import numpy as np
import wandb


class Dataset(Dataset):
    def __init__(self, datapath):
        super(Dataset, self).__init__()
        self.datapath = datapath
        self.image_names = []
        for image_name in os.listdir(datapath):
            self.image_names.append(image_name)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, i):
        image = read_image(os.path.join(self.datapath, self.image_names[i]))
        image = Resize((128, 128))(image)
        image = image/255   # sprowadzamy instensywność pikseli do zakresu 0-1
        return image

class Generator(nn.Module):
    def __init__(self, n_features):
        super(Generator, self).__init__()
        # im_size = (im_size-1)*stride - 2*padding + kernel_size
        self.convt1 = nn.ConvTranspose2d(n_features, 512, kernel_size=4, stride=1, padding=0) # 1x1x200 -> 4x4x512
        self.convt2 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)   # 8x8x256
        self.convt3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)   # 16x16x128
        self.convt4 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)    # 32x32x64
        self.convt5 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)     # 64x64x32
        self.convt6 = nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1)      # 128x128x3
        self.norm1 = nn.BatchNorm2d(512)
        self.norm2 = nn.BatchNorm2d(256)
        self.norm3 = nn.BatchNorm2d(128)
        self.norm4 = nn.BatchNorm2d(64)
        self.norm5 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, noise_vector):
        y = self.relu(self.norm1(self.convt1(noise_vector)))
        y = self.relu(self.norm2(self.convt2(y)))
        y = self.relu(self.norm3(self.convt3(y)))
        y = self.relu(self.norm4(self.convt4(y)))
        y = self.relu(self.norm5(self.convt5(y)))
        y = self.tanh(self.convt6(y))
        return y


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # im_size = (im_size + 2*padding - kernel_size)/stride + 1
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)   # 128x128x3 -> 64x64x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)   # 32x32x32
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)   # 16x16x64
        self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)   # 8x8x128
        self.conv5 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)   # 4x4x256
        self.conv6 = nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0)   # 1x1x1
        self.norm1 = nn.InstanceNorm2d(16)
        self.norm2 = nn.InstanceNorm2d(32)
        self.norm3 = nn.InstanceNorm2d(64)
        self.norm4 = nn.InstanceNorm2d(128)
        self.norm5 = nn.InstanceNorm2d(256)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, im_batch):
        y = self.lrelu(self.norm1(self.conv1(im_batch)))
        y = self.lrelu(self.norm2(self.conv2(y)))
        y = self.lrelu(self.norm3(self.conv3(y)))
        y = self.lrelu(self.norm4(self.conv4(y)))
        y = self.lrelu(self.norm5(self.conv5(y)))
        y = self.conv6(y)
        return y


class Critic2(nn.Module):
    def __init__(self, h_dim=16):
        super(Critic2, self).__init__()

        # n_new = (n + 2*pad -ker)/stride + 1
        self.crit = nn.Sequential(
            nn.Conv2d(3, h_dim, kernel_size=4, stride=2, padding=1),    # 128x128x3 -> 64x64x16
            nn.InstanceNorm2d(h_dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(h_dim, h_dim*2, kernel_size=4, stride=2, padding=1),  # 32x32x32
            nn.InstanceNorm2d(h_dim * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(h_dim * 2, h_dim * 4, kernel_size=4, stride=2, padding=1),  # 16x16x64
            nn.InstanceNorm2d(h_dim * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(h_dim * 4, h_dim * 8, kernel_size=4, stride=2, padding=1),  # 8x8x128`
            nn.InstanceNorm2d(h_dim * 8),
            nn.LeakyReLU(0.2),

            nn.Conv2d(h_dim * 8, h_dim * 16, kernel_size=4, stride=2, padding=1),  # 4x4x256
            nn.InstanceNorm2d(h_dim * 16),
            nn.LeakyReLU(0.2),

            nn.Conv2d(h_dim * 16, 1, kernel_size=4, stride=1, padding=0),  # 1x1x1
        )

    def forward(self, image):
        pred = self.crit(image)             # 128x1x1x1
        return pred.view(len(pred), -1)     # 128x1


def generate_noise(n_features, batch_size):
    return torch.randn(batch_size, n_features, 1, 1)


def gradient_penalty(real_imgs, fake_imgs, lambd=10):
    alfa = torch.rand(real_imgs.shape[0], 1, 1, 1, device=device, requires_grad=True)
    mixed_imgs = alfa * real_imgs + (1 - alfa) * fake_imgs
    mixed_predictions = critic(mixed_imgs)
    gradients = torch.autograd.grad(outputs=mixed_predictions, inputs=mixed_imgs,
                                    grad_outputs=torch.ones_like(mixed_predictions),
                                    retain_graph=True, create_graph=True)[0]
    gradients_norms = gradients.norm(p=2)
    return lambd * ((gradients_norms - 1)**2).mean()


def wandb_img_send(img_tensor):
    img_tensor.permute(1, 2, 0).clip(0, 1)
    wandb.log({'image':wandb.Image()})

# hyperparameters
batch_size = 128
n_noise_vector_features = 200
crit_to_gen_training_ratio = 5
lr = 0.01
n_epochs = 100

# Ogarnianie wandba
wandbactive = True  # if we want to track stats with wandb
wandb.login(key='2f2183383168b5071375fc9091a49ddef158fb27')
experiment_name = wandb.util.generate_id()
myrun = wandb.init(project='advGAN', group=experiment_name,
                   config={
                       'dataset':'Spitfires30-30240',
                       'optimizer':'Adam',
                       'model':'wgan gp',
                       'epochs':n_epochs,
                       'batch_size':batch_size
                   })
config = wandb.config

# parameters
datapath = './dataset'
datapath = "./data/celeba/img_align_celeba"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = Dataset(datapath)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(n_features=n_noise_vector_features)
critic = Critic()
generator.to(device)
critic.to(device)

gen_optim = torch.optim.Adam(generator.parameters(), lr=lr)
crit_optim = torch.optim.Adam(critic.parameters(), lr=lr)

for epoch in range(n_epochs):
    for real_batch in dataloader:
        # critic
        mean_critic_loss = 0
        for _ in range(crit_to_gen_training_ratio):
            # dodać no_grady
            crit_optim.zero_grad()

            noise_vector = generate_noise(n_noise_vector_features, batch_size)
            fake_batch = generator(noise_vector).detach()

            real_predictions = critic(real_batch)
            fake_predictions = critic(fake_batch)

            gp = gradient_penalty(real_batch, fake_batch)
            crit_loss = fake_predictions.mean() - real_predictions.mean() + gp
            mean_critic_loss += crit_loss/crit_to_gen_training_ratio     # line for mean critic loss calculation, not impacts training process
            crit_loss.backward()

            crit_optim.step()


        # generator
        gen_optim.zero_grad()

        noise_vector = generate_noise(n_noise_vector_features, batch_size)
        fake_batch = generator(noise_vector)

        # nie trzeba z detachować krytyka?
        fake_predictions = critic(fake_batch)

        gen_loss = fake_predictions.mean()
        gen_loss.backward()

        gen_optim.step()

        # wandb result show
        # statystyki
        if wandbactive:
            wandb.log({'Epoch': epoch, 'Critic loss': mean_critic_loss, 'Generator loss': gen_loss})


    # wysyłamy obrazki na koniec epoki
    wandb_img_send(fake_batch[0])
