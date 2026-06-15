

import torchvision.transforms as T
import torch
import imageio
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import einops
import imageio
import random
from torch.optim.lr_scheduler import StepLR
import torch

# DDPM class
class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)

        # Linear noise schedule - ensure all tensors are on the correct device
        self.beta = torch.linspace(min_beta, max_beta, n_steps, device=device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        self.sigma = torch.sqrt(self.beta)

    def get_alpha(self, t):
        """Get alpha value for a given timestep t"""
        # Ensure t is on the same device as self.alpha
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device).long()
        else:
            t = t.to(self.device).long()
        return self.alpha[t]

    def get_alpha_bar(self, t):
        """Get alpha_bar value for a given timestep t"""
        # Ensure t is on the same device as self.alpha_bar
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device).long()
        else:
            t = t.to(self.device).long()
        return self.alpha_bar[t]

    def get_beta(self, t):
        """Get beta value for a given timestep t"""
        # Ensure t is on the same device as self.beta
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device).long()
        else:
            t = t.to(self.device).long()
        return self.beta[t]

    def get_sigma(self, t):
        """Get sigma value for a given timestep t"""
        # Ensure t is on the same device as self.sigma
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device).long()
        else:
            t = t.to(self.device).long()
        return self.sigma[t]

    def to(self, device):
        """Override to method to ensure all tensors are moved to the correct device"""
        super().to(device)
        self.device = device
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)
        self.sigma = self.sigma.to(device)
        return self

    def forward(self, x0, t, eta=None, mask=None):
        """
        Applies diffusion noise only to the masked regions.
        - x0: Original (clean) image.
        - t: Timestep.
        - eta: Noise (optional).
        - mask: Binary mask (1 = keep pixel, 0 = masked/missing).
        """
        n, c, h, w = x0.shape
        a_bar = self.get_alpha_bar(t)

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)  # Generate noise

        if mask is None:
            mask = torch.zeros_like(x0).to(self.device)  # Default: No mask (apply to whole image)

        # Apply noise only to masked regions
        noisy = mask * x0 + (1 - mask) * (self.get_alpha(t).sqrt().reshape(n, 1, 1, 1) * x0 + (1 - self.get_alpha(t)).sqrt().reshape(n, 1, 1, 1) * eta)

        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)

    def sample(self, x, t):
        """
        Sample from the model given a noisy input x at timestep t.
        This is the reverse diffusion process.
        """
        # Convert t to tensor if it's not already
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=self.device).long()
        else:
            # Ensure t is on the correct device
            t = t.to(self.device).long()
        
        # Ensure t has the correct shape
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        # Predict the noise
        eta_theta = self.backward(x, t)
        
        # Get the alpha values for this timestep and ensure they're on the correct device
        alpha_t = self.get_alpha(t).to(x.device)
        alpha_t_bar = self.get_alpha_bar(t).to(x.device)
        
        # Compute the mean of the reverse process
        mean = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)
        
        # If we're at the last timestep, we don't need to add noise
        if t[0].item() == 0:
            return mean
            
        # Add noise for all other timesteps
        z = torch.randn_like(x)
        sigma_t = self.get_sigma(t).to(x.device)
        return mean + sigma_t * z


def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=None, h=224, w=224):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    frame_idxs = np.linspace(0, ddpm.n_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device
        
        # Use the number of channels from the model's image_chw
        if c is None:
            c = ddpm.image_chw[0]

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        for idx, t in enumerate(list(range(ddpm.n_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.get_alpha(t)
            alpha_t_bar = ddpm.get_alpha_bar(t)

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.get_beta(t)
                sigma_t = beta_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples ** 0.5))
                frame = frame.cpu().numpy().astype(np.uint8)

                # Rendering frame
                frames.append(frame)

        # Storing the GIF
        with imageio.get_writer(gif_name, mode="I") as writer:
            for idx, frame in enumerate(frames):
                frame = np.array(frame)  # Ensure it's a NumPy array

                # Check if the frame is grayscale (1 channel) or already RGB (3 channels)
                if len(frame.shape) == 2:  # Grayscale image (H, W)
                    frame = np.stack([frame] * 3, axis=-1)  # Convert to (H, W, 3)
                elif frame.shape[-1] == 1:  # Single-channel (H, W, 1)
                    frame = np.squeeze(frame, axis=-1)  # Remove the last dim (H, W)
                    frame = np.stack([frame] * 3, axis=-1)  # Convert to (H, W, 3)

                writer.append_data(frame)  # Save frame

                # Save the last frame multiple times for looping effect
                if idx == len(frames) - 1:
                    for _ in range(frames_per_gif // 3):
                        writer.append_data(frame)  # Use converted RGB frame

    return x


def show_forward(ddpm, loader, device):
    # Showing the forward process
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, "Original images")

        for percent in [0.25, 0.5, 0.75, 1]:
            show_images(
                ddpm(imgs.to(device),
                     [int(percent * ddpm.n_steps) - 1 for _ in range(len(imgs))]),
                f"DDPM Noisy images {int(percent * 100)}%"
            )
        break

class MyBlock(nn.Module):
    def __init__(self, shape, in_c, out_c, kernel_size=3, stride=1, padding=1, activation=None, normalize=True):
        super(MyBlock, self).__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size, stride, padding)
        self.activation = nn.SiLU() if activation is None else activation
        self.normalize = normalize

    def forward(self, x):
        out = self.ln(x) if self.normalize else x
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.activation(out)
        return out
    
def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding

def _make_te(self, dim_in, dim_out):
  return nn.Sequential(
    nn.Linear(dim_in, dim_out),
    nn.SiLU(),
    nn.Linear(dim_out, dim_out)
  )



class MyUNet_32(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100):
        super(MyUNet_32, self).__init__()

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((1, 28, 28), 1, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10)
        )
        self.down1 = nn.Conv2d(10, 10, 4, 2, 1)

        self.te2 = self._make_te(time_emb_dim, 10)
        self.b2 = nn.Sequential(
            MyBlock((10, 14, 14), 10, 20),
            MyBlock((20, 14, 14), 20, 20),
            MyBlock((20, 14, 14), 20, 20)
        )
        self.down2 = nn.Conv2d(20, 20, 4, 2, 1)

        self.te3 = self._make_te(time_emb_dim, 20)
        self.b3 = nn.Sequential(
            MyBlock((20, 7, 7), 20, 40),
            MyBlock((40, 7, 7), 40, 40),
            MyBlock((40, 7, 7), 40, 40)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(40, 40, 2, 1),
            nn.SiLU(),
            nn.Conv2d(40, 40, 4, 2, 1)
        )

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 40)
        self.b_mid = nn.Sequential(
            MyBlock((40, 3, 3), 40, 20),
            MyBlock((20, 3, 3), 20, 20),
            MyBlock((20, 3, 3), 20, 40)
        )

        # Second half
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(40, 40, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(40, 40, 2, 1)
        )

        self.te4 = self._make_te(time_emb_dim, 80)
        self.b4 = nn.Sequential(
            MyBlock((80, 7, 7), 80, 40),
            MyBlock((40, 7, 7), 40, 20),
            MyBlock((20, 7, 7), 20, 20)
        )

        self.up2 = nn.ConvTranspose2d(20, 20, 4, 2, 1)
        self.te5 = self._make_te(time_emb_dim, 40)
        self.b5 = nn.Sequential(
            MyBlock((40, 14, 14), 40, 20),
            MyBlock((20, 14, 14), 20, 10),
            MyBlock((10, 14, 14), 10, 10)
        )

        self.up3 = nn.ConvTranspose2d(10, 10, 4, 2, 1)
        self.te_out = self._make_te(time_emb_dim, 20)
        self.b_out = nn.Sequential(
            MyBlock((20, 28, 28), 20, 10),
            MyBlock((10, 28, 28), 10, 10),
            MyBlock((10, 28, 28), 10, 10, normalize=False)
        )

        self.conv_out = nn.Conv2d(10, 1, 3, 1, 1)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))  # (N, 10, 28, 28)
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))  # (N, 20, 14, 14)
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))  # (N, 40, 7, 7)

        out_mid = self.b_mid(self.down3(out3) + self.te_mid(t).reshape(n, -1, 1, 1))  # (N, 40, 3, 3)

        out4 = torch.cat((out3, self.up1(out_mid)), dim=1)  # (N, 80, 7, 7)
        out4 = self.b4(out4 + self.te4(t).reshape(n, -1, 1, 1))  # (N, 20, 7, 7)

        out5 = torch.cat((out2, self.up2(out4)), dim=1)  # (N, 40, 14, 14)
        out5 = self.b5(out5 + self.te5(t).reshape(n, -1, 1, 1))  # (N, 10, 14, 14)

        out = torch.cat((out1, self.up3(out5)), dim=1)  # (N, 20, 28, 28)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))  # (N, 1, 28, 28)

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )

class MyUNet(nn.Module):
    def __init__(self, n_steps=1000, time_emb_dim=100, in_channels=3, out_channels=3):
        super(MyUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # Downsampling Path (Encoder)
        self.te1 = self._make_te(time_emb_dim, 1)
        self.b1 = nn.Sequential(
            MyBlock((in_channels, 224, 224), in_channels, 32),
            MyBlock((32, 224, 224), 32, 32),
        )
        self.down1 = nn.Conv2d(32, 64, 4, 2, 1)  # 224 → 112

        self.te2 = self._make_te(time_emb_dim, 64)
        self.b2 = nn.Sequential(
            MyBlock((64, 112, 112), 64, 64),
            MyBlock((64, 112, 112), 64, 64),
        )
        self.down2 = nn.Conv2d(64, 128, 4, 2, 1)  # 112 → 56

        self.te3 = self._make_te(time_emb_dim, 128)
        self.b3 = nn.Sequential(
            MyBlock((128, 56, 56), 128, 128),
            MyBlock((128, 56, 56), 128, 128),
        )
        self.down3 = nn.Conv2d(128, 256, 4, 2, 1)  # 56 → 28

        self.te4 = self._make_te(time_emb_dim, 256)
        self.b4 = nn.Sequential(
            MyBlock((256, 28, 28), 256, 256),
            MyBlock((256, 28, 28), 256, 256),
        )
        self.down4 = nn.Conv2d(256, 512, 4, 2, 1)  # 28 → 14

        self.te5 = self._make_te(time_emb_dim, 512)
        self.b5 = nn.Sequential(
            MyBlock((512, 14, 14), 512, 512),
            MyBlock((512, 14, 14), 512, 512),
        )
        self.down5 = nn.Conv2d(512, 512, 4, 2, 1)  # 14 → 7

        # Bottleneck
        self.te_mid = self._make_te(time_emb_dim, 512)
        self.b_mid = nn.Sequential(
            MyBlock((512, 7, 7), 512, 512),
            MyBlock((512, 7, 7), 512, 512),
        )

        # Upsampling Path (Decoder)
        self.up1 = nn.ConvTranspose2d(512, 512, 4, 2, 1)  # 7 → 14
        self.te6 = self._make_te(time_emb_dim, 1024)
        self.b6 = nn.Sequential(
            MyBlock((1024, 14, 14), 1024, 512),
            MyBlock((512, 14, 14), 512, 256),
        )

        self.up2 = nn.ConvTranspose2d(256, 256, 4, 2, 1)  # 14 → 28
        self.te7 = self._make_te(time_emb_dim, 512)
        self.b7 = nn.Sequential(
            MyBlock((512, 28, 28), 512, 256),
            MyBlock((256, 28, 28), 256, 128),
        )

        self.up3 = nn.ConvTranspose2d(128, 128, 4, 2, 1)  # 28 → 56
        self.te8 = self._make_te(time_emb_dim, 256)
        self.b8 = nn.Sequential(
            MyBlock((256, 56, 56), 256, 128),
            MyBlock((128, 56, 56), 128, 64),
        )

        self.up4 = nn.ConvTranspose2d(64, 64, 4, 2, 1)  # 56 → 112
        self.te9 = self._make_te(time_emb_dim, 128)
        self.b9 = nn.Sequential(
            MyBlock((128, 112, 112), 128, 64),
            MyBlock((64, 112, 112), 64, 32),
        )

        self.up5 = nn.ConvTranspose2d(32, 32, 4, 2, 1)  # 112 → 224
        self.te_out = self._make_te(time_emb_dim, 64)
        self.b_out = nn.Sequential(
            MyBlock((64, 224, 224), 64, 32),
            MyBlock((32, 224, 224), 32, 32, normalize=False)
        )

        self.conv_out = nn.Conv2d(32, out_channels, 3, 1, 1)

    def to(self, device):
        """Override to method to ensure time_embed is moved to the correct device"""
        super().to(device)
        self.time_embed = self.time_embed.to(device)
        return self

    def forward(self, x, t):
        # Ensure t is on the correct device
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x.device).long()
        else:
            t = t.to(x.device).long()
            
        t = self.time_embed(t)
        n = len(x)

        # Rest of the forward pass remains the same
        out1 = self.b1(x + self.te1(t).reshape(n, -1, 1, 1))
        out2 = self.b2(self.down1(out1) + self.te2(t).reshape(n, -1, 1, 1))
        out3 = self.b3(self.down2(out2) + self.te3(t).reshape(n, -1, 1, 1))
        out4 = self.b4(self.down3(out3) + self.te4(t).reshape(n, -1, 1, 1))
        out5 = self.b5(self.down4(out4) + self.te5(t).reshape(n, -1, 1, 1))

        out_mid = self.b_mid(self.down5(out5) + self.te_mid(t).reshape(n, -1, 1, 1))

        out6 = torch.cat((out5, self.up1(out_mid)), dim=1)
        out6 = self.b6(out6 + self.te6(t).reshape(n, -1, 1, 1))

        out7 = torch.cat((out4, self.up2(out6)), dim=1)
        out7 = self.b7(out7 + self.te7(t).reshape(n, -1, 1, 1))

        out8 = torch.cat((out3, self.up3(out7)), dim=1)
        out8 = self.b8(out8 + self.te8(t).reshape(n, -1, 1, 1))

        out9 = torch.cat((out2, self.up4(out8)), dim=1)
        out9 = self.b9(out9 + self.te9(t).reshape(n, -1, 1, 1))

        out = torch.cat((out1, self.up5(out9)), dim=1)
        out = self.b_out(out + self.te_out(t).reshape(n, -1, 1, 1))

        out = self.conv_out(out)

        return out

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )



def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    rows = int(len(images) ** (1 / 2))
    cols = round(len(images) / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    plt.show()
    

