import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Resize
import os
import numpy as np
import wandb


class Dataset(Dataset):
    def __init__(self, datapath):
        super(Dataset, self).__init__()
        self.datapath = datapath
        self.image_names = []
        for image_name in os.listdir(datapath)[:10000]:
            self.image_names.append(image_name)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, i):
        image = read_image(os.path.join(self.datapath, self.image_names[i]))
        image = Resize((128, 72))(image)
        image = image / 255  # sprowadzamy instensywność pikseli do zakresu 0-1
        return image


class Generator(nn.Module):
    def __init__(self, n_features):
        super(Generator, self).__init__()
        # im_size = (im_size-1)*stride - 2*padding + kernel_size
        self.gen = nn.Sequential(
            self.gen_block(n_features, 512, stride=1, padding=0),  # 1x1x200 -> 4x4x512
            self.gen_block(512, 256, stride=2, padding=1),  # 8x8x256
            self.gen_block(256, 128, stride=2, padding=1),  # 16x16x128
            self.gen_block(128, 64, stride=2, padding=1),  # 32x32x64
            self.gen_block(64, 32, stride=2, padding=1),  # 64x64x32
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),  # 128x128x3
            nn.Tanh()
        )

    def gen_block(self, input_size, output_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(input_size, output_size, kernel_size=4,
                               stride=stride, padding=padding),
            nn.BatchNorm2d(output_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, noise_vector):
        return self.gen(noise_vector)


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        # im_size = (im_size + 2*padding - kernel_size)/stride + 1
        self.crit = nn.Sequential(
            self.crit_block(3, 16),  # 128x128x3 -> 64x64x16
            self.crit_block(16, 32),  # 32x32x32
            self.crit_block(32, 64),  # 16x16x64
            self.crit_block(64, 128),  # 8x8x128
            self.crit_block(128, 256),  # 4x4x256
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=0)  # 1x1x1
        )

    def crit_block(self, input_size, output_size):
        return nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(output_size),
            nn.LeakyReLU(0.2)
        )

    def forward(self, im_batch):
        return self.crit(im_batch)


def generate_noise(n_features, batch_size, device):
    return torch.randn(batch_size, n_features, 1, 1, device=device)


def gradient_penalty(real_imgs, fake_imgs, critic, lambd=10):
    alfa = torch.rand(real_imgs.shape[0], 1, 1, 1, device=device, requires_grad=True)
    mixed_imgs = alfa * real_imgs + (1 - alfa) * fake_imgs
    mixed_predictions = critic(mixed_imgs)
    gradients = torch.autograd.grad(outputs=mixed_predictions, inputs=mixed_imgs,
                                    grad_outputs=torch.ones_like(mixed_predictions),
                                    retain_graph=True, create_graph=True)[0]
    gradients = gradients.view(gradients.shape[0], -1)  # 128x49152
    gradients_norms = gradients.norm(p=2, dim=1)
    return lambd * ((gradients_norms - 1) ** 2).mean()


def wandb_img_send(imgs_tensor):
    # take first two images from batch
    img_tensor1 = imgs_tensor[0].detach().cpu()
    img_tensor2 = imgs_tensor[1].detach().cpu()
    # permute from torch CxWxH to normal WxHxC
    # and clip intensities from [-1; 1] to [0; 1]
    img_tensor1 = img_tensor1.permute(1, 2, 0).clip(0, 1)
    img_tensor2 = img_tensor2.permute(1, 2, 0).clip(0, 1)
    # transform to np arays to make it possible to send to wandb
    img1 = np.array(img_tensor1)
    img2 = np.array(img_tensor2)
    wandb.log({'image1': wandb.Image(img1), 'image2': wandb.Image(img2)})


# hyperparameters
batch_size = 128
n_noise_vector_features = 200
crit_to_gen_training_ratio = 5
gen_lr = 0.0003
crit_lr = 0.0003
n_epochs = 50


# Ogarnianie wandba
wandbactive = True  # if we want to track stats with wandb
wandb.login(key='2f2183383168b5071375fc9091a49ddef158fb27')
experiment_name = wandb.util.generate_id()
myrun = wandb.init(project='advGAN', group=experiment_name,
                   config={
                       'dataset': 'Spitfires30-30240',
                       'optimizer': 'Adam',
                       'model': 'wgan gp',
                       'epochs': n_epochs,
                       'batch_size': batch_size,
                       'genarator_lr': gen_lr,
                       'critic_lr': crit_lr
                   })
config = wandb.config

# parameters
datapath = './data/spitfires_mltpl'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# torch.cuda.empty_cache()
step = 0

print('Ladowanie datasetu...')
dataset = Dataset(datapath)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print('Dataset załadowany!')

generator = Generator(n_features=n_noise_vector_features).to(device)
critic = Critic().to(device)

gen_optim = torch.optim.Adam(generator.parameters(), lr=crit_lr, betas=(0.5, 0.9))
crit_optim = torch.optim.Adam(critic.parameters(), lr=crit_lr, betas=(0.5, 0.9))

for epoch in range(n_epochs):
    for real_batch in dataloader:
        real_batch = real_batch.to(device)
        cur_batch_size = len(real_batch)
        # critic
        mean_critic_loss = 0
        for _ in range(crit_to_gen_training_ratio):
            crit_optim.zero_grad()

            noise_vector = generate_noise(n_noise_vector_features,
                                          cur_batch_size, device)
            fake_batch = generator(noise_vector).detach()

            fake_predictions = critic(fake_batch)
            real_predictions = critic(real_batch)

            gp = gradient_penalty(real_batch, fake_batch, critic)
            crit_loss = fake_predictions.mean() - real_predictions.mean() + gp
            mean_critic_loss += (
                        crit_loss / crit_to_gen_training_ratio).item()  # line for mean critic loss calculation, not impacts training process
            crit_loss.backward()

            crit_optim.step()

        # generator
        gen_optim.zero_grad()

        noise_vector = generate_noise(n_noise_vector_features, cur_batch_size,
                                      device)
        fake_batch = generator(noise_vector)

        # nie trzeba z detachować krytyka?
        fake_predictions = critic(fake_batch)

        gen_loss = - fake_predictions.mean()
        gen_loss.backward()

        gen_optim.step()

        print('step', step, 'gen loss:', gen_loss.item(),
              'critic loss:', mean_critic_loss)
        # wandb result show

        # statystyki
        if wandbactive:
            wandb.log({
                'Epoch': epoch,
                'Critic loss': mean_critic_loss,
                'Step': step,
                'Generator loss': gen_loss.item()
            })

        step += 1

    # wysyłamy obrazki na koniec epoki
    wandb_img_send(fake_batch)
