import os
import numpy as np
from loguru import logger
import wandb
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGAugmentor:
    def __init__(self,
                 strategy='amplitude_noise',
                 batch=False,
                 ratio_cutout=0.5,
                 single=False):
        self.ratio_scale = 1.2
        self.ratio_noise = 0.05
        self.ratio_cutout = ratio_cutout
        self.ratio_time_shift = 0.1  # Shift 10% of time
        self.batch = batch

        self.aug = True
        if strategy == '' or strategy.lower() == 'none':
            self.aug = False
        else:
            self.strategy = []
            for aug in strategy.lower().split('_'):
                self.strategy.append(aug)
        logger.info('Augmentation strategy = {}'.format(self.strategy))
        self.aug_fn = {
            'amplitude': [self.amplitude_scaling],
            'noise': [self.noise_injection],
            # 'timemask': [self.time_masking],
            # 'electrodeshuffle': [self.electrode_shuffle],
            'timeshift': [self.time_shift],
        }

    def __call__(self, x, single_aug=True, seed=-1):
        if not self.aug:
            return x
        else:
            if len(self.strategy) > 0:
                if single_aug:
                    # Apply one random augmentation
                    idx = np.random.randint(len(self.strategy))
                    p = self.strategy[idx]
                    for f in self.aug_fn[p]:
                        self.set_seed(seed)
                        x = f(x, self.batch)
                else:
                    # Apply all selected augmentations
                    for p in self.strategy:
                        for f in self.aug_fn[p]:
                            self.set_seed(seed)
                            x = f(x, self.batch)

            x = x.contiguous()
            return x

    def set_seed(self, seed):
        if seed > 0:
            np.random.seed(seed)
            torch.random.manual_seed(seed)

    def amplitude_scaling(self, x, batch=True):
        """Scales EEG amplitude similar to contrast augmentation."""
        ratio = self.ratio_scale
        if batch:
            scale = np.random.uniform(1 / ratio, ratio)
            x = x * scale
        else:
            scale = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * (ratio - 1 / ratio) + 1 / ratio
            x = x * scale
        return x

    def noise_injection(self, x, batch=True):
        """Adds Gaussian noise to EEG signals."""
        noise_std = self.ratio_noise * x.std()
        noise = torch.randn_like(x) * noise_std
        return x + noise

    # def time_masking(self, x, batch=True):
    #     """Randomly masks a section of the time dimension."""
    #     mask_size = int(x.size(2) * self.ratio_cutout)  # Cut out % of time samples
    #     if batch:
    #         start = np.random.randint(0, x.size(2) - mask_size)
    #     else:
    #         start = torch.randint(0, x.size(2) - mask_size, (x.size(0),), device=x.device)
    #     mask = torch.ones_like(x)
    #     mask[:, :, start:start + mask_size] = 0
    #     return x * mask

    # def electrode_shuffle(self, x, batch=True):
    #     """Shuffles electrode order randomly."""
    #     if batch:
    #         perm = torch.randperm(x.size(1), device=x.device)
    #         return x[:, perm, :]
    #     else:
    #         perm = torch.stack([torch.randperm(x.size(1), device=x.device) for _ in range(x.size(0))])
    #         return x.gather(1, perm.unsqueeze(-1).expand_as(x))

    def time_shift(self, x, batch=True):
        """Shifts EEG signals along the time axis."""
        shift_amount = int(x.size(2) * self.ratio_time_shift)
        if batch:
            shift = np.random.randint(-shift_amount, shift_amount + 1)
            return torch.roll(x, shifts=shift, dims=2)
        else:
            shifts = torch.randint(-shift_amount, shift_amount + 1, (x.size(0),), device=x.device)
            return torch.stack([torch.roll(x[i], shifts=int(shifts[i]), dims=1) for i in range(x.size(0))])


class RBF(nn.Module):
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth_multipliers = self.bandwidth_multipliers.cuda()
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)
        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(-L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers)[:, None, None]).sum(dim=0)


class M3DLoss(nn.Module):

    def __init__(self, kernel_type: str = 'gaussian'):
        super().__init__()
        if kernel_type == 'gaussian':
            self.kernel = RBF()
        elif kernel_type == 'linear':
            self.kernel = LinearKernel()
        elif kernel_type == 'polinominal':
            self.kernel = PoliKernel()
        elif kernel_type == 'laplace':
            self.kernel = LaplaceKernel()

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))
        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY


def save_eeg(save_dir, eeg_data, file_prefix: str = '', num_wandb_samples=3):
    """
    Save EEG signals as images and log sample images to WANDB.

    :param save_dir: str, directory to save the EEG images.
    :param eeg_data: Tensor, EEG data of shape (batch, 1, channels, time).
    :param file_prefix: str, prefix for saved files.
    :param num_wandb_samples: int, number of sample EEG visualizations to log to WANDB.
    """
    os.makedirs(save_dir, exist_ok=True)

    eeg_data = eeg_data.cpu().numpy()

    # Save the EEG data as a NumPy file
    export_path = os.path.join(save_dir, f'{file_prefix}.npy')
    np.save(export_path, eeg_data)
    logger.debug(f'Synthesized EEG (shape={eeg_data.shape}) saved to {export_path}')

    # Select random samples to visualize
    batch_size, _, channel_num, time_len = eeg_data.shape
    sample_indices = np.random.choice(batch_size, num_wandb_samples, replace=False)

    wandb_images = []

    for idx in sample_indices:
        sample = np.squeeze(eeg_data[idx], axis=0)  # shape (62, 200)

        # Create subplots: 8 x 8 grid (max 64) for 62 channels
        fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(16, 16))
        axes = axes.flatten()

        for ch in range(channel_num):
            ax = axes[ch]
            ax.plot(sample[ch])
            ax.set_title(f"Ch {ch}")
            ax.set_xticks([])
            ax.set_yticks([])

        # Hide any unused subplots
        for ax in axes[channel_num:]:
            ax.axis('off')

        fig.suptitle(f"EEG Segment Index: {idx}", fontsize=16)
        plt.tight_layout()

        # Save the figure to an image buffer and log it to WANDB
        image_path = os.path.join(save_dir, f"{file_prefix}_sample_{idx}.png")
        fig.savefig(image_path)
        plt.close(fig)  # Prevents showing multiple figures

        wandb_images.append(wandb.Image(image_path, caption=f"EEG Sample {idx}"))

    # Log images to WANDB
    wandb.log({f"EEG_Sample_Images_{file_prefix}": wandb_images})


if __name__ == '__main__':
    dummy_eeg = torch.randn([256, 1, 62, 200])

    # augmentor = EEGAugmentor()
    # while True:
    #     augmentor(dummy_eeg)
    save_eeg('.', dummy_eeg)
