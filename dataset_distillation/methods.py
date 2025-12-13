"""
Distillation methods

@Author: Tang Yuting
@Date: 17 Feb 2025
"""

import re
import os
import time
import random

try:
    from typing_extensions import Literal, List, Union
except ImportError:
    from typing import Literal, List, Union

import wandb
from loguru import logger
from tqdm import tqdm
import joblib

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import svds

from sklearn.metrics import pairwise_distances
# ----- torch related -----
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torcheeg.models.cnn import EEGNet
# -------------------------
from sklearn.decomposition import PCA
from scipy.sparse.linalg import svds
from sklearn.decomposition import TruncatedSVD, IncrementalPCA
from sklearn.cluster import KMeans

from pyod.models.hbos import HBOS
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer

try:
    from .distil_utils import *
    from .kcenter_greedy import kCenterGreedy
    from .cross_domain_transformer import CrossReconstructionTransformer
    from .cross_domain_transformer.anomaly_transformer import AnomalyTransformer
    from .deep_infomax import *
except ImportError as e:
    logger.warning(e)
    from distil_utils import *
    from kcenter_greedy import kCenterGreedy
    from cross_domain_transformer import CrossReconstructionTransformer
    from cross_domain_transformer.anomaly_transformer import AnomalyTransformer
    from deep_infomax import *

IS_DEBUG = False


def normalize_mat(X):
    """
    :param X: np.array, the feature matrix of each sample point with the shape of (n_sample, n_channel, n_time)
    :return: normalized matrix with the same shape as X
    """
    # Log the initial properties of X
    # logger.debug(f"normalize_mat: Input shape: {X.shape}, dtype: {X.dtype}, "
    #              f"min: {X.min().item()}, max: {X.max().item()}")
    sample_num = X.shape[0]
    flattened_X = X.reshape(sample_num, -1)  # this line is checked such that reshaping is performing in row-order
    # logger.debug(f"normalize_mat: flattened_X shape: {flattened_X.shape}")

    mean_values = torch.mean(flattened_X, dim=1)
    std_values = torch.std(flattened_X, dim=1)
    # logger.debug(f"normalize_mat: mean_values shape: {mean_values.shape}, first 5 means: {mean_values[:5]}")
    # logger.debug(f"normalize_mat: std_values shape: {std_values.shape}, first 5 stds: {std_values[:5]}")

    mean_values = mean_values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    std_values = std_values.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    # logger.debug(f"normalize_mat: mean_values shape after unsqueeze: {mean_values.shape}")
    # logger.debug(f"normalize_mat: std_values shape after unsqueeze: {std_values.shape}")

    eps = 1e-8
    normalized_X = (X - mean_values) / (std_values + eps)
    # logger.debug(f"normalize_mat: normalized_X shape: {normalized_X.shape}, "
    #              f"min: {normalized_X.min().item()}, max: {normalized_X.max().item()}")
    return normalized_X


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class RandomDistillation:
    """Randomly selects a subset of data per class."""

    def apply(self, dataset, labels, reduction_ratio, seed=42, is_sample_by_class: bool = True, **kwargs):
        random.seed(seed)
        np.random.seed(seed)

        distilled_data, distilled_labels = [], []
        if is_sample_by_class:
            unique_labels = set(labels)

            # Log class label distribution before distillation
            original_distribution = {label: sum(1 for lbl in labels if lbl == label) for label in unique_labels}
            logger.info(f"Original class distribution: {original_distribution}")

            for label in unique_labels:
                indices = [i for i, lbl in enumerate(labels) if lbl == label]
                selected_indices = random.sample(indices, int(len(indices) * reduction_ratio))
                distilled_data.extend([dataset[i] for i in selected_indices])
                distilled_labels.extend([labels[i] for i in selected_indices])

            # Log class label distribution after distillation
            new_distribution = {label: sum(1 for lbl in distilled_labels if lbl == label) for label in unique_labels}
            logger.info(f"New class distribution after distillation: {new_distribution}")
        else:  # e.g., regression
            selected_indices = np.random.randint(0, len(dataset), size=int(len(dataset) * reduction_ratio))

            distilled_data = dataset[selected_indices, ...]
            distilled_labels = labels[selected_indices, ...]

        return np.array(distilled_data), np.array(distilled_labels)


class M3DSyntheticDistillation:
    def __init__(self, features, labels, reduction_ratio, class2initial_size: dict = None, device='cuda',
                 is_debug: bool = False, loss_type: str = 'm3d', **kwargs):
        """
        :param features: Tensor of EEG data (real dataset)
        :param labels: Tensor of corresponding class labels
        :param class2initial_size: dict, e.g., {0: 5140, 1: 5140, 2: 5140}
        :param reduction_ratio: float, factor to reduce synthesizer size.
        """
        self.device = device
        self.class2size = class2initial_size or {}  # Store class size mapping
        self.reduction_ratio = reduction_ratio
        self.is_debug = is_debug
        if self.is_debug:
            logger.info('Using debugging mode so that wandb is not initialized.')
        if loss_type == 'm3d':
            self.criterion = M3DLoss()
        else:
            self.criterion = nn.MSELoss()
        logger.info('The loss function used = {}'.format(self.criterion))

        self.model_kwargs = {'chunk_size': 200, 'num_electrodes': 62, 'F1': 4, 'F2': 8, 'num_classes': 32}
        logger.info('The arguments of the model = {}'.format(self.model_kwargs))

        # Track Best Synthetic Dataset
        self.best_loss = float('inf')  # Initialize as infinity
        self.best_data = None
        self.best_labels = None

        # Adjust synthesizer size based on reduction ratio
        synthesizer_size = int(sum(self.class2size.values()) * reduction_ratio)

        self.__init_synthetic_set(features, synthesizer_size, labels)

        self.augmentor = EEGAugmentor()

    def __init_synthetic_set(self, features: Union[torch.Tensor, DataLoader], synthesizer_size, labels=None):
        """
        Initialize the synthetic dataset with real EEG segments
        """
        # Infer synthesizer_size if not provided
        if synthesizer_size <= 0:
            if isinstance(features, torch.utils.data.DataLoader):
                dataset_len = len(features.dataset)
            elif isinstance(features, torch.Tensor):
                dataset_len = features.size(0)
            elif isinstance(features, (np.ndarray, list)):
                dataset_len = len(features)
            else:
                raise ValueError("Unsupported type for `features` when inferring synthesizer_size.")

            synthesizer_size = int(dataset_len * self.reduction_ratio)

        # ----- Deal with different input types -----
        if isinstance(features, torch.utils.data.DataLoader):
            all_features = []
            all_labels = [] if labels is not None else None
            accumulated_samples = 0

            for batch in tqdm(features, desc='Iterating over "features", which is a DataLoader'):
                if labels is None:
                    x = batch
                else:
                    x, y = batch
                    all_labels.append(y)
                all_features.append(x)

                # Estimate sample count and early stop condition
                batch_size = x[0].size(0) if isinstance(x, (tuple, list)) else x.size(0)
                accumulated_samples += batch_size

                # Stop if more than 1.5 * synthesizer_size, otherwise OOM
                if accumulated_samples > int(synthesizer_size * 1.5):
                    logger.info(
                            f"Early stopping: loaded {accumulated_samples} samples which exceeds 150% of synthesizer_size={synthesizer_size}")
                    break

            if isinstance(all_features[0], (list, tuple)):
                features = torch.cat([x[0] for x in all_features], dim=0)
            else:
                features = torch.cat([x for x in all_features], dim=0)
            logger.debug('features.shape={}'.format(features.shape))
            if labels is not None:
                labels = torch.cat(all_labels, dim=0)

        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features, dtype=torch.float32, )
            labels = torch.tensor(labels, dtype=torch.long, ) if labels is not None else None
        _, channel_num, time_len = features.shape

        if features.ndim == 3:
            features = features.unsqueeze(1)
        synthetic_shape = (synthesizer_size, 1, channel_num, time_len)
        self.model_kwargs['chunk_size'] = time_len
        self.model_kwargs['num_electrodes'] = channel_num
        # ----- ----- -----

        logger.info(
                'Initializing the synthesized dataset (shape={}) with {}% real EEG segments...'
                .format(synthetic_shape, self.reduction_ratio * 100))
        class2real_data = {c: features[labels == c] for c in self.class2size.keys()} if labels is not None else None
        # Create synthetic dataset using real EEG segments
        with torch.no_grad():  # Prevent gradient tracking during initialization
            self.data = torch.nn.Parameter(torch.randn(synthetic_shape, dtype=torch.float32))
            self.targets = torch.zeros(synthesizer_size, dtype=torch.long, )
            if labels is None:
                logger.info("No labels provided; initializing without class-based selection.")
                sampled_indices = torch.randperm(features.size(0))[:synthesizer_size]
                real_samples = features[sampled_indices]
                logger.debug("Normalizing the real_samples")
                real_samples = normalize_mat(real_samples.detach())
                logger.debug("Finish normalization")
                self.data[:real_samples.size(0)] = real_samples.clone().detach()
                self.targets[:real_samples.size(0)] = 0  # Single-class fallback
                self.class_indices = {0: list(range(real_samples.size(0)))}
            else:
                start_idx = 0
                self.class_indices = {}
                for cls, size in self.class2size.items():
                    reduced_size = int(size * self.reduction_ratio)  # Adjust for reduction ratio
                    real_samples = class2real_data[cls][:reduced_size]  # Take required samples per class

                    if len(real_samples) < reduced_size:
                        raise ValueError(
                                f"Not enough real EEG samples for class {cls}. Required: {reduced_size}, "
                                f"Available: {len(real_samples)}")
                    real_samples = normalize_mat(real_samples)
                    # Use a new tensor instead of in-place modification
                    self.data[start_idx:start_idx + reduced_size] = real_samples.clone().detach()

                    self.class_indices[cls] = list(range(start_idx, start_idx + reduced_size))
                    self.targets[start_idx: start_idx + reduced_size] = cls
                    start_idx += reduced_size
        logger.info("Finish initialization")

    def sample(self, n, cls: int = None):
        """
        Sample `n` examples from `self.data` within class `cls`, ensuring correct targets.

        :param cls: int or None — class label to sample from, or None for full-data sampling.
        :param n: int, number of samples to draw.
        :return: (Tensor of shape (n, 1, 62, 200), Tensor of shape (n,))
        """
        if cls is None:
            total_available = self.data.shape[0]
            if total_available < n:
                logger.warning(
                        f"Requested {n} samples from class {cls}, but only {total_available} available. Returning all.")
                n = total_available

            sampled_indices = torch.randperm(total_available)[:n]
        else:
            if cls not in self.class_indices:
                logger.warning(f"Class {cls} not found in dataset. Returning empty tensors.")
                return self.data[:0], self.targets[:0]

            indices = self.class_indices[cls]
            if len(indices) < n:
                logger.warning(
                        f"Requested {n} samples from class {cls}, but only {len(indices)} available. Returning all.")
                n = len(indices)

            sampled_indices = torch.tensor(indices)[torch.randperm(len(indices))[:n]]

        return self.data[sampled_indices], self.targets[sampled_indices]

    def apply(self, features, labels, batch_size=256, iteration_num=5, learning_rate=0.001, momentum=0.05):
        """
        :return: np.array(distilled_data), np.array(distilled_labels)
        """
        # Adjust the batch_size to avoid error
        batch_size = int(
                min(self.class2size.values()) * self.reduction_ratio // 2) if len(self.class2size) > 0 else 64

        # Initialize wandb
        if not self.is_debug:
            hyperparam_dict = {"reduction_ratio": self.reduction_ratio,
                               'learning_rate': learning_rate,
                               'momentum': momentum,
                               'batch_size': batch_size}
            wandb.init(project="LaBraM", entity='tangyuti', group='M3D',
                       config={**hyperparam_dict, **self.model_kwargs},
                       tags=['distill', 'init_real'])

        ##### Save the initial synthesized data #####
        for c in self.class2size.keys():
            sample_data, _ = self.sample(c, 32)
            save_eeg('.', sample_data.cpu().detach(), file_prefix=f'class{c}_iteration-1')
        #########################

        if isinstance(features, DataLoader):
            class2subset_loader = features
        else:
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features)
                labels = torch.tensor(labels, dtype=torch.long)
            dataset = TensorDataset(features, labels)
            class2subset_loader = {c: [data for data in dataset if data[1] == c] for c in self.class2size.keys()}

        for it in tqdm(range(iteration_num), desc='Iteratively updating the synthetic set'):
            self.__iterate(batch_size, class2subset_loader, it, learning_rate)
            if IS_DEBUG:
                break

        # Finish wandb logging
        wandb.finish()

        # Return the best synthetic dataset found
        if labels is None:
            return np.array(self.best_data.numpy())
        else:
            return np.array(self.best_data.numpy()), np.array(self.best_labels.numpy())

    def __iterate(self, batch_size, class2subset_or_loader, it, learning_rate):
        if it % 1 == 0:  # ipm=5
            model = EEGNet(**self.model_kwargs).to(self.device)
            model.train()
        optim_img = torch.optim.Adam([self.data], lr=learning_rate)
        loss_total = 0
        batch_counter = 0
        self.data.data = torch.clamp(self.data.data, min=-1., max=1.)

        is_unsupervised = isinstance(class2subset_or_loader, DataLoader)

        if is_unsupervised:
            loader = class2subset_or_loader
            class_loss_total = 0
            for img in tqdm(loader, desc='Iterating unlabeled data'):
                if isinstance(img, (list, tuple)):
                    img = img[0]  # Discard labels if present
                if img.ndim == 3:
                    img = img.unsqueeze(1).to(self.device).float()
                img_syn, _ = self.sample(batch_size, None)  # Unsupervised sample
                img_syn = self.augmentor(img_syn.to(self.device))
                img = self.augmentor(normalize_mat(img.to(self.device)))

                with torch.no_grad():
                    feat_tg = model(img)
                feat = model(img_syn)

                loss = self.criterion(feat.mean(0), feat_tg.mean(0)) if isinstance(self.criterion, nn.MSELoss) \
                    else self.criterion(feat, feat_tg)

                loss_total += loss.item()
                class_loss_total += loss.item()
                batch_counter += 1

                optim_img.zero_grad()
                loss.backward()
                optim_img.step()

                if IS_DEBUG:
                    break

            wandb.log({"Loss_Unlabeled": class_loss_total / len(loader)})
        else:
            for c in self.class2size.keys():
                class_loader = DataLoader(class2subset[c], batch_size=batch_size, shuffle=True)
                class_loss_total = 0  # Track loss per class

                for img, _ in tqdm(class_loader, desc=f'Iterating class {c}'):
                    img_syn, _ = self.sample(c, batch_size)
                    img_syn = img_syn.to(self.device)
                    img = img.to(self.device)
                    img = normalize_mat(img)
                    img = self.augmentor(img)  # img.shape=256, 1, 62, 200
                    img_syn = self.augmentor(img_syn)

                    with torch.no_grad():
                        feat_tg = model(img)
                    feat = model(img_syn)
                    if isinstance(self.criterion, nn.MSELoss):
                        # Compute mean of feature representations
                        mean_real = feat_tg.mean(dim=0)
                        mean_syn = feat.mean(dim=0)
                        loss = self.criterion(mean_syn, mean_real)
                    else:
                        loss = self.criterion(feat, feat_tg)

                    loss_total += loss.item()
                    class_loss_total += loss.item()
                    batch_counter += 1

                    optim_img.zero_grad()
                    loss.backward()
                    optim_img.step()

                # Log class-level loss
                wandb.log({f"Loss_Class_{c}": class_loss_total / len(class_loader)})
        # Log average loss per iteration
        average_loss = loss_total / batch_counter
        wandb.log({"Iteration": it, "Average_Loss": average_loss})
        # Update the best dataset if current loss is lower
        if average_loss < self.best_loss:
            logger.info(f'The best model found at Iteration {it}')
            self.best_loss = average_loss
            self.best_data = self.data.clone().detach().cpu()
            self.best_labels = self.targets.clone().detach().cpu()

        ##### Save the sample synthesized data #####
        if is_unsupervised:
            sample_data, _ = self.sample(32, None)
            save_eeg('.', sample_data.cpu().detach(), file_prefix=f'unlabeled_iteration{it}')
        else:
            for c in self.class2size.keys():
                sample_data, _ = self.sample(32, c)
                save_eeg('.', sample_data.cpu().detach(), file_prefix=f'class{c}_iteration{it}')
        ###############


def compute_parameter_variance(model):
    """
    Computes the sum of variances of all parameters in the model.
    If you want the average variance, divide by total number of params, etc.
    """
    var_sum = 0.0
    count = 0
    for p in model.parameters():
        # Only consider parameters that require grad
        if p.requires_grad:
            # Flatten the parameter and compute variance
            param_data = p.detach().view(-1)
            var_sum += param_data.var().item()
            count += 1
    return var_sum / count if count > 0 else 0.0


def compute_gradient_variance(model):
    grad_var_sum = 0.0
    grad_count = 0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.view(-1)
            grad_var_sum += g.var().item()
            grad_count += 1
    return grad_var_sum / grad_count if grad_count > 0 else 0.0


def generate_magnitude_phase_info(time):
    """
    To generate a feature matrix, including time, phase and magnitude information.
    :param time: torch.Tensor, with the dimension (SAMPLE_NUM, CHANNEL_NUM, TIMESTAMPS)
    """
    # Computes the one dimensional discrete Fourier transform of input.
    freq = torch.fft.fft(time, dim=-1)[:, :, :time.shape[-1] // 2]

    a = freq.real
    b = freq.imag
    magnitude = torch.abs(freq)
    phase = torch.zeros_like(a)
    phase[a > 0] = torch.atan(b / a)[a > 0]
    phase[a < 0] = (torch.atan(b / a) + torch.sign(b) * np.pi)[a < 0]
    phase[a == 0] = (torch.sign(b) * np.pi / 2)[a == 0]
    return time, magnitude, phase


class KCenterDistillation:
    def __init__(self, reduction_ratio, contamination: float = 0.15, distance_metric='euclidean',
                 feature_extraction: Literal['SVD', 'SSL'] = 'SVD',
                 device='cuda', use_vae=False, ssl_epoch_num=50,
                 ssl_type: Literal['CRT', 'VQAE', 'DeepInfomax'] = 'CRT',
                 is_quantize: bool = False, compression_ratios: List[int] = None,
                 ssl_kwargs: dict = None, batch_size: int = 64, cache_path: str = '',
                 contamination_method: Literal['HBOS', 'MC'] = 'HBOS', is_export_indices: bool = False, **kwargs):
        """
        Initialize the k-Center Distillation method.

        :param model: Pretrained feature extractor (e.g., VGG-16)
        :param distance_metric: Distance metric ('Euclidean' or 'cosine')
        :param device: 'cuda' or 'cpu'
        :param contamination: float in (0., 0.5)
        """
        self.feature_extraction = feature_extraction
        self.device = device if not IS_DEBUG else 'cpu'
        self.distance_metric = distance_metric
        self.reduction_ratio = reduction_ratio
        self.contamination = contamination
        self.batch_size = batch_size
        self.use_vae = use_vae
        self.ssl_epoch_num = ssl_epoch_num
        self.is_export_indices = is_export_indices
        self.contamination_method = contamination_method
        self.channel_num = kwargs['channel_num'] if 'channel_num' in kwargs else None
        self.model_name_suffix = kwargs['model_name_suffix'] if 'model_name_suffix' in kwargs else ''
        self.metadata = kwargs['metadata'] if 'metadata' in kwargs else None
        self.ssl_model = None  # VAE model instance
        self.ssl_kwargs = {
                'seq_len': 400,
                'patch_len': 10,
                'dim': 64,
                'num_class': 2,
                'in_dim': self.channel_num,
                'channel_num': self.channel_num,
                } if ssl_kwargs is None else ssl_kwargs
        if 'is_anomaly' in kwargs:
            self.ssl_kwargs['is_anomaly'] = kwargs['is_anomaly']
        self.ssl_type = ssl_type
        self.cache_path = cache_path
        self.is_quantize = False
        self.is_only_ssl = kwargs['is_only_ssl'] if 'is_only_ssl' in kwargs else False
        if compression_ratios is None:
            self.compression_ratios = [2, 5, 5, 2]
        else:
            self.compression_ratios = compression_ratios

        self.global_discriminator = Discriminator().to(self.device)
        self.local_discriminator = Discriminator().to(self.device)

        logger.info("The architecture {} will be used for SSL.".format(self.ssl_type))

    def train_ssl(self, dataset_or_dataloader, latent_dim=64, learning_rate=0.001, mask_ratio=0.5):
        """
        Train the model on the dataset in a self-supervised manner.

        :param dataset_or_dataloader: Either a PyTorch Dataset or a DataLoader.
                                  If it is a Dataset, the batch dimension could be 3 or 4.
        :param latent_dim: Dimensionality of latent representation
        :param learning_rate: Learning rate for training
        :param mask_ratio: Mask ratio for the SSL model
        :return: Trained SSL model
        """
        # If already a DataLoader, use it directly; otherwise create a DataLoader from the given Dataset
        if isinstance(dataset_or_dataloader, DataLoader):
            # Recreate DataLoader with the same dataset but shuffle enabled
            dataloader = DataLoader(
                    dataset_or_dataloader.dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True
                    )
        elif isinstance(dataset_or_dataloader, np.ndarray):
            dataloader = DataLoader(
                    TensorDataset(torch.tensor(dataset_or_dataloader)),
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True  # avoid a batch with 1 sample, which might cause errors in IDC loss
                    )
        else:
            dataloader = DataLoader(
                    dataset_or_dataloader,
                    batch_size=self.batch_size,
                    shuffle=True,
                    drop_last=True  # avoid a batch with 1 sample, which might cause errors in IDC loss
                    )

        if not IS_DEBUG:
            wandb.init(project="LaBraM", entity="tangyuti", group="SSL",
                       config={"latent_dim": latent_dim, "learning_rate": learning_rate, "mask_ratio": mask_ratio,
                               "model": self.ssl_type, "dataset": self.model_name_suffix},
                       tags=["distill", "SSL", self.ssl_type])
        self.__train_crt(dataloader, learning_rate, mask_ratio)
        if self.cache_path != '':
            state_dict = torch.load(self.cache_path, map_location=torch.device(self.device))
            logger.info("Loading SSL model from {}".format(self.cache_path))
            self.ssl_model.load_state_dict(state_dict)
            self.ssl_model.eval()
            return self.ssl_model

        # ---------------------------
        # 1) Save the final model locally:
        model_save_path = "sslmodel_{}.pth".format(self.model_name_suffix)
        torch.save(self.ssl_model.state_dict(), model_save_path)
        # 2) Log the saved model file to wandb:
        try:
            wandb.save(model_save_path)
            logger.info("The trained model is saved to {}".format(model_save_path))
        except Exception as e:
            logger.warning(f"The model cannot be saved due to the error {e}")
        # ---------------------------

        logger.info("SSL Training Completed.")
        try:
            wandb.finish()
        except Exception as e:
            logger.warning("wandb.finish() fails due to the error {e}")

        return self.ssl_model

    def __train_crt(self, dataloader, learning_rate, mask_ratio):
        # Initialize SSL model
        logger.debug("Initializing CRT with parameters {}".format(self.ssl_kwargs))
        self.ssl_model = CrossReconstructionTransformer(**self.ssl_kwargs).to(self.device)
        pytorch_total_params = sum(p.numel() for p in self.ssl_model.parameters() if p.requires_grad)
        logger.info(f'Parameters={pytorch_total_params}')
        if self.cache_path != '':
            return
        optimizer = optim.Adam(self.ssl_model.parameters(), lr=learning_rate)
        # Example improvement: Use a learning-rate scheduler, e.g. StepLR
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        logger.info(f"Training SSL with mask_ratio={mask_ratio}")
        global_step = 0
        for epoch in tqdm(range(self.ssl_epoch_num if not IS_DEBUG else 1), desc="Training SSL"):
            epoch_loss = 0.0
            # We'll keep a running_loss to log every 5 steps
            running_loss = 0.0

            for batch_idx, batch in tqdm(enumerate(dataloader, start=1)):
                global_step, loss = self.__forward_ssl_model(batch, global_step)
                optimizer.zero_grad()

                loss.backward()

                # Example improvement: Gradient clipping to prevent exploding grads
                torch.nn.utils.clip_grad_norm_(self.ssl_model.parameters(), max_norm=5.0)

                optimizer.step()

                epoch_loss += loss.item()
                running_loss += loss.item()

                # Log running loss every 5 steps
                if (batch_idx % 5) == 0:
                    grad_variance = compute_gradient_variance(self.ssl_model)
                    if not IS_DEBUG:
                        wandb.log(
                                {
                                        "Train Step": global_step,
                                        "Loss/Rec": loss.item(),
                                        "Gradient Variance": grad_variance
                                        }
                                )
                    running_loss = 0.0

                if IS_DEBUG:
                    break

            # Accumulate the epoch-level statistics
            avg_epoch_loss = epoch_loss / len(dataloader)  # Average loss
            param_variance = compute_parameter_variance(self.ssl_model)

            # Step the scheduler after each epoch
            scheduler.step()

            # Log epoch-level metrics
            if not IS_DEBUG:
                wandb.log(
                        {
                                "Epoch": epoch + 1,
                                "Epoch Loss": avg_epoch_loss,
                                "Learning Rate": optimizer.param_groups[0]["lr"],
                                "Parameter Variance": param_variance,
                                }
                        )

            logger.info(f"[Epoch {epoch + 1}] Loss: {avg_epoch_loss:.4f}")

    def __forward_ssl_model(self, batch, global_step, is_ssl=True, forward_num: int = 1):
        global_step += 1
        if isinstance(batch, list):
            batch = batch[0]
        batch = batch.float().to(self.device)
        batch = normalize_mat(batch.unsqueeze(1) if batch.ndim == 3 else batch)
        # Example improvement: Preprocessing in dataset __getitem__
        # rather than per batch can be more efficient. But here’s the same:
        potential, magnitude, phase = generate_magnitude_phase_info(batch.squeeze(1))
        magnitude = normalize_mat(magnitude.unsqueeze(1) if magnitude.ndim == 3 else magnitude).squeeze(1)
        phase = normalize_mat(phase.unsqueeze(1) if phase.ndim == 3 else phase).squeeze(1)
        batch = torch.concat([potential, magnitude, phase], axis=-1)
        if is_ssl:
            loss, association_dict = self.ssl_model(batch, ratio=0, ssl=True)
            return global_step, loss
        else:
            if forward_num == 1:
                features = self.ssl_model(batch, ssl=False)
                return features
            else:
                embeddings_list = []
                for _ in range(forward_num):
                    embeddings = self.ssl_model(batch, ssl=False)
                    embeddings_list.append(embeddings)
                # Shape: (τ, B, D)
                stacked_embeddings = torch.stack(embeddings_list)
                return stacked_embeddings

    def encode(self, dataset, batch_size=256):
        """
        After `train_ssl`, call this to retrieve encoded representations of the entire dataset.
        """
        if not self.ssl_model:
            raise RuntimeError("ssl_model is not trained or initialized yet.")

        logger.info("Encoding the dataset with batch_size={}".format(batch_size))
        self.ssl_model.eval()
        self.ssl_model.to(self.device)
        if isinstance(dataset, DataLoader):
            dataloader = torch.utils.data.DataLoader(dataset.dataset, batch_size=dataset.batch_size, shuffle=False,
                                                     drop_last=False)  # Create a new loader without shuffling
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_features = []
        all_subject_idx = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                if isinstance(batch, (list, tuple)):
                    batch, subject_idx = batch
                features = self.__forward_ssl_model(batch, batch_idx, is_ssl=False)

                all_features.append(features.cpu())
                if subject_idx is not None:
                    all_subject_idx.append(subject_idx.cpu())
                if IS_DEBUG:
                    break

        # Concatenate all features
        all_features = torch.cat(all_features, dim=0)
        if len(all_subject_idx) > 0:
            all_subject_idx = torch.cat(all_subject_idx, dim=0)
        elif self.metadata is not None:  # for seed
            all_subject_idx = self.metadata['subject_id'].values
        if not IS_DEBUG:
            export_path = '{}_ssl_features.pt'.format(int(time.time()))
            torch.save(all_features, export_path)
            logger.info(f"Encoded dataset size: {all_features.shape}\tSaved into {export_path}")
        return all_features, all_subject_idx

    def compute_ood_scores_mc_dropout(self, dataset, num_inferences=10, batch_size=256):
        """
        Estimate OOD scores for a dataset using Monte Carlo Dropout.
        OOD score is defined as the variance of the model's output probability for the positive class.

        Args:
            dataset: a torch.utils.data.Dataset or DataLoader
            num_inferences: number of stochastic forward passes
            batch_size: batch size for inference

        Returns:
            variance across inferences as OOD scores
        """
        if not self.ssl_model:
            raise RuntimeError("ssl_model is not trained or initialized yet.")

        # ----- Set up the model for dropout -----
        self.ssl_model.eval()
        self.ssl_model.to(self.device)
        # Enable dropout during test time
        def enable_dropout(m):
            for each_module in m.modules():
                if each_module.__class__.__name__.startswith('Dropout'):
                    each_module.train()
        enable_dropout(self.ssl_model)
        # ----- ----- -----

        # Prepare data loader
        if isinstance(dataset, torch.utils.data.DataLoader):
            dataloader = torch.utils.data.DataLoader(dataset.dataset, batch_size=dataset.batch_size, shuffle=False)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

        uncertainties = []
        with torch.no_grad():
            for batch_idx, batch_ in enumerate(dataloader):
                stacked_embeddings = self.__forward_ssl_model(batch_, batch_idx, is_ssl=False, forward_num=20)
                var_per_dim = torch.var(stacked_embeddings, dim=0)
                ood_score = var_per_dim.mean(dim=-1)
                uncertainties.append(ood_score.cpu())

        return torch.cat(uncertainties, dim=0)

    def apply(self, dataset, labels, k=32, seed=42, metric='euclidean'):
        """
        Distill the dataset by selecting the most representative samples.
        :return: Distilled dataset (subset of original data)
        """
        if isinstance(dataset, DataLoader):
            sample_num = len(dataset.dataset)
        else:
            sample_num, *_ = dataset.shape
        features, subject_indices = self.reduce_dim(dataset, k, sample_num)
        if self.is_only_ssl:
            return None

        original_indices = np.arange(features.shape[0])
        if self.contamination > 0:
            replacement = f"{self.contamination_method}{str(self.contamination).replace('.', '')}"
            # Use regex to replace the part after 'sslmodel_' up to '.pth'
            dirpath, filename = os.path.split(self.cache_path)
            stem, ext = os.path.splitext(filename)
            new_stem = re.sub(r'([^/]+model)', replacement, stem)
            outlier_label_path = new_stem + '.npy'
            if os.path.exists(outlier_label_path):
                logger.info(f"Loading cached outlier labels from {outlier_label_path}")
                outlier_labels = np.load(outlier_label_path)
            else:
                if self.contamination_method == 'HBOS':
                    # ----- Get anomaly scores and predictions -----
                    feature_matrix_np = standardizer(features)

                    ood_detector = HBOS(n_bins='auto', contamination=self.contamination)
                    # ood_detector = LOF(contamination=contamination)
                    logger.info(f"Calculating Outlier Score with {ood_detector} (contamination={self.contamination})")

                    ood_detector.fit(feature_matrix_np)
                    # 1 = Outlier, 0 = not Outlier
                    outlier_labels = ood_detector.predict(feature_matrix_np, return_confidence=False).astype(bool)
                else:  # MC dropout
                    ood_scores = self.compute_ood_scores_mc_dropout(dataset, batch_size=256)
                    # given the scores, sort and remove the top x% samples with highest OOD scores
                    num_to_remove = int(self.contamination * len(ood_scores))
                    indices_sorted = torch.argsort(ood_scores)  # ascending order
                    indices_to_remove = indices_sorted[-num_to_remove:].detach().cpu().numpy()
                    logger.debug(
                            f"Removing {num_to_remove}/{len(ood_scores)} most uncertain samples (top {self.contamination:.2%})")
                    outlier_labels = np.zeros(len(ood_scores), dtype=bool)
                    outlier_labels[indices_to_remove] = True
                np.save(outlier_label_path, outlier_labels)
                logger.info(f"Outlier labels saved to {outlier_label_path}")

            logger.debug(f"Before OOD removal, features.shape={features.shape}")
            features = features[~outlier_labels]
            subject_indices = subject_indices[~outlier_labels]
            original_indices = original_indices[~outlier_labels]
            if labels is not None:
                labels = labels[~outlier_labels]
            logger.debug(f"After OOD, features.shape={features.shape}")
        # ----- Perform k-center -----
        if not IS_DEBUG:
            num_centers = int(sample_num * self.reduction_ratio)  # Adjust based on the required number of centers
            solver = kCenterGreedy(features, labels, seed, metric)  # Initialize k-Center Greedy solver
            selected_indices = solver.select_batch_(model=None, already_selected=[], N=num_centers)
            # Output the selected indices
            logger.debug(f"Selected indices: {selected_indices}")

            # Log subject indices of selected samples
            if subject_indices is not None:
                selected_subjects = subject_indices[selected_indices]
                original_indices = original_indices[selected_indices]
                log_lines = ["Selected samples with subject mapping:"]
                for i, subj, origin_i in zip(selected_indices, selected_subjects, original_indices):
                    log_lines.append(f"  Index: {i}, Subject ID: {subj}, Original Index: {origin_i}")
                logger.debug("\n".join(log_lines))
        else:
            selected_indices = np.arange(len(features))[:100]
        # ----- ----- -----

        if self.is_export_indices:
            logger.warning('Skipping selected samples as self.is_export_indices=True')
            return None
        else:   # Retrieve selected samples
            if isinstance(dataset, DataLoader):
                full_dataset = dataset.dataset
                # Use the selected indices to manually retrieve the samples
                subset_data = [full_dataset[idx] for idx in selected_indices]
                if isinstance(subset_data[0], (list, tuple)):  # when dataset returns eeg_segment, subject_idx
                    subset_data = [x[0] for x in subset_data]
                subset_data = np.stack(subset_data)
            else:
                subset_data = dataset[selected_indices]

            if labels is not None:
                subset_labels = labels[selected_indices]
                return subset_data, subset_labels
            else:
                return subset_data

    def reduce_dim(self, dataset, k, sample_num):
        """
        :return: features
        """
        if self.feature_extraction in ['SSL', 'DeepInfomax']:  # Train and extract VAE representations if enabled
            if not isinstance(dataset, DataLoader):
                dataset_shape = dataset.shape
                logger.info(f'Training Self-supervised learning on the array (shape={dataset_shape})')
            start_time = time.time()
            self.train_ssl(dataset)
            features, subject_indices = self.encode(dataset)
            logger.debug("Shape of EEG features:", features.shape)
            end_time = time.time()
            logger.debug(f"SSL execution time: {end_time - start_time:.4f} seconds")
        else:
            if isinstance(dataset, Dataset):
                dataset_shape = dataset.shape
            elif isinstance(dataset, np.ndarray):
                dataset_shape = dataset.shape
            else:
                dataset_shape = dataset.dataset.feature_size
            logger.info(f'Apply Incremental SVD on the array (shape={dataset_shape})')
            start_time = time.time()

            # ----- Load the SVD model -----
            model_save_path = "svdmodel_{}.pth".format(self.model_name_suffix)

            if self.cache_path == '':  # when trained model is not used
                batch_size, svd = self.__train_svd(dataset, k, model_save_path)
            else:
                if os.path.exists(model_save_path):
                    svd = joblib.load(model_save_path)
                    logger.info("Loaded pretrained IncrementalPCA from disk.")
                else:
                    logger.warning('Training SVD since {} does not exist'.format(model_save_path))
                    batch_size, svd = self.__train_svd(dataset, k, model_save_path)
            # ----- ----- -----

            # Optionally transform dataset into reduced features
            features = []
            subject_indices = []
            if isinstance(dataset, np.ndarray):
                for i in tqdm(range(0, dataset.shape[0], batch_size), desc='Transforming dataset'):  # for SEED
                    end_idx = min(i + batch_size, dataset.shape[0])
                    batch = dataset[i:end_idx]
                    batch_np = batch.reshape(batch.shape[0], -1)
                    features.append(svd.transform(batch_np))
            else:
                for batch, subject_idx in tqdm(dataset, desc='Transforming dataset'):  # for LaBraM
                    batch_np = batch.numpy().reshape(batch.shape[0], -1)
                    features.append(svd.transform(batch_np))
                    subject_indices.append(subject_idx.numpy().reshape(subject_idx.shape[0], -10))
            features = np.vstack(features)
            subject_indices = np.vstack(subject_indices)

            logger.debug("Shape of reduced EEG features: {}".format(features.shape))
            end_time = time.time()
            logger.debug(f"SVD execution time: {end_time - start_time:.4f} seconds")

        return features, subject_indices

    def __train_svd(self, dataset, k, model_save_path):
        svd = IncrementalPCA(n_components=k)
        batch_size = 512  # adjust based on your memory budget
        logger.info('Training SVD {} with batch size {}'.format(svd, batch_size))

        partial_data = []
        if isinstance(dataset, np.ndarray):  # iterate over a numpy when dataset is very large
            for i in tqdm(range(0, dataset.shape[0], batch_size), desc='Fitting SVD in chunks'):
                end_idx = min(i + batch_size, dataset.shape[0])
                batch = dataset[i:end_idx]
                batch_np = batch.reshape(batch.shape[0], -1)
                svd.partial_fit(batch_np)
        else:
            for batch, _ in tqdm(dataset, desc='Fitting SVD in batches'):
                batch_np = batch.numpy().reshape(batch.shape[0], -1)
                svd.partial_fit(batch_np)  # Only accumulates the components
        logger.debug(f"Saving the trained {svd} to {model_save_path}")
        joblib.dump(svd, model_save_path)
        return batch_size, svd


if __name__ == "__main__":
    # Initialize and Apply k-Center Distillation
    # # ----- VQAE -----
    # center_distillation = KCenterDistillation(feature_extraction='SSL', reduction_ratio=0.5,
    #                                           distance_metric='euclidean', device='cuda', channel_num=64,
    #                                           ssl_epoch_num=1, ssl_type='VQAE', is_discard_ood=True)
    # # ----- ----- -----

    # # ----- CRT -----
    # center_distillation = KCenterDistillation(feature_extraction='SSL', reduction_ratio=0.5,
    #                                           distance_metric='euclidean', device='cuda', channel_num=23,
    #                                           ssl_epoch_num=1, ssl_type='CRT', is_discard_ood=True)
    # # ----- ----- -----s

    # # ----- AnomalyTransformer -----
    # center_distillation = KCenterDistillation(feature_extraction='SSL', reduction_ratio=0.5,
    #                                           distance_metric='euclidean', device='cuda', channel_num=23,
    #                                           ssl_epoch_num=1, ssl_type='AnomalyTransformer', is_discard_ood=True,
    #                                           ssl_kwargs={'win_size': 2200, 'patch_len': 10, 'channel_num': 23,
    #                                                       'd_model': 64})
    # # ----- ----- -----

    # ----- Deep Infomax -----
    # Generate Random Dataset (Example)
    num_samples = 1000
    data = np.random.rand(num_samples, 23, 200)  # 64 to follow CRT
    labels = np.random.randint(0, 3, num_samples)  # Simulated labels

    center_distillation = KCenterDistillation(feature_extraction='SSL', reduction_ratio=0.5,
                                              distance_metric='euclidean', device='cuda', channel_num=23,
                                              ssl_epoch_num=1, ssl_type='DeepInfomax', is_discard_ood=True,
                                              ssl_kwargs={'seq_len': 200 * 2, 'patch_len': 10, 'in_dim': 23,
                                                          'num_class': 2, 'dim': 64, 'channel_num': 23})
    # ----- ----- -----

    distilled_data, distilled_labels = center_distillation.apply(data, labels)

    print(f"Distilled Dataset Size: {len(distilled_data)} samples")
