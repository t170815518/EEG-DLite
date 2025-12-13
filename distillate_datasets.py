"""
The script to distillate the pretraining datasets and save them to local .npy files
@Author: Tang Yuting
@Date: 2025-March-16
"""

# ----- Important packages -----
import os
import time
import fire
import h5py
import traceback
from typing import List
from tqdm.auto import tqdm
from loguru import logger
from pathlib import Path

import torch
import numpy as np

from utils import build_pretraining_dataset
from data_processor.dataset import SingleShockDataset  # Assuming this is defined elsewhere
from dataset_distillation import *  # Using the distillation framework

# ----- ----- -----


DATASET_NUM = 33
BIGGEST_SIZE_ALLOWED = float('+inf')


def main(distillation_method="m3d",  # Choose between 'random', 'm3d', 'coreset'
         reduction_ratio=0.5,
         base_dir="/home/yuting/data/h5data/",
         feature_extraction: str = 'SSL',  # 'SSL', 'SVD'
         cache_path: str = '',
         export_path="/home/yuting/distilled_h5data",
         specific_datasets: str = None,
         is_only_ssl: bool = False,
         is_only_kcenter: bool = False,
         contamination: float = 0,
         contamination_method: str = 'HBOS'):
    """

    :param distillation_method:
    :param reduction_ratio:
    :param base_dir:
    :param feature_extraction:
    :param cache_path: str, '' means no cache model is used
    :param export_path:
    :return:
    """
    # Set up the log file
    logfile_name = 'distillate_dataset_{}.log'.format(int(time.time()))
    logger.add(logfile_name, level="DEBUG", format="{time} {level} {message}")
    logger.info("distillation_method={}, reduction_ratio={}, base_dir={}, feature_extraction={}".format(
            distillation_method, reduction_ratio, base_dir, feature_extraction))

    output_dir = "{}_{}{}{}{}/".format(
            export_path, distillation_method,
            '' if feature_extraction == 'SSL' else feature_extraction,
            str(int(reduction_ratio * 100)),
            '' if contamination == 0 else '_{}{}'.format(contamination_method, str(contamination).replace('.', '')))

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    logger.info('OUTPUT_DIR={}'.format(output_dir))

    logger.info("Building pretraining datasets")
    dataset_paths = [os.path.join(base_dir, x) for x in os.listdir(base_dir)]
    dataset_train_list, train_ch_names_list = build_pretraining_dataset(dataset_paths, time_window=None,
                                                                        stride_size=800, start_percentage=0,
                                                                        end_percentage=1, sample_percentage=1.0,
                                                                        is_return_subject_id=True)
    prog_bar = tqdm(total=DATASET_NUM)

    for shockdataset in dataset_train_list:
        single_datasets = shockdataset.get_datasets()
        for singleshock_dataset in single_datasets:
            if len(singleshock_dataset) > BIGGEST_SIZE_ALLOWED:
                continue

            if specific_datasets is not None:
                if singleshock_dataset.get_filepath().stem != specific_datasets:
                    logger.warning(
                            f'Dataset {singleshock_dataset.get_filepath().stem} not in specific_datasets, so it is skipped')
                    continue

            cache_path_ = '' if cache_path == '' else (
                    os.path.join(cache_path, 'sslmodel_{}.pth'.format(singleshock_dataset.get_filepath().stem)))
            if is_only_kcenter and not os.path.exists(cache_path_):
                raise FileNotFoundError(f'Unable to perform k-center only since {cache_path_} does not exist.')

            try:
                distillate_dataset(singleshock_dataset, output_dir, distillation_method, reduction_ratio, cache_path_,
                                   feature_extraction, is_only_ssl=is_only_ssl, contamination=contamination,
                                   contamination_method=contamination_method)
            except FileNotFoundError as e:
                logger.warning('Skipping {} due to FileNotFoundError {}'.format(cache_path_, e))
                logger.error(traceback.format_exc())
                continue
            except ValueError as e:
                logger.warning('Skipping {} due to ValueError {}'.format(cache_path_, e))
                logger.error(traceback.format_exc())
                continue

    logger.info("✅ Dataset distillation completed.")


def distillate_dataset(dataset, output_dir, distillation_method, reduction_ratio, cache_path, feature_extraction='SSL',
                       ssl_type='DeepInfomax', is_only_ssl: bool = False, contamination: float = 0,
                       contamination_method='HBOS'):
    logger.info(f"Processing: {dataset.get_filepath()}")
    dataset_name = dataset.get_filepath().stem
    distilled_file_path = Path(output_dir) / (dataset_name + '.npy')

    if os.path.exists(distilled_file_path):
        logger.info("{} exists so skipping".format(distilled_file_path))
        return

    if len(dataset) > BIGGEST_SIZE_ALLOWED:
        return

    # # ----- Extract data and labels (Don't use this snippet for very large datasets otherwise out of memory!) -----
    # data_list, label_list = [], []
    # for i in tqdm(range(len(dataset))):
    #     sample = dataset[i]  # Assuming dataset[i] returns (data, label)
    #     if sample.shape[1] == dataset.feature_size[1]:
    #         data_list.append(sample)
    #     else:
    #         logger.warning("Skip one sample (shape={}) which differs from {}".format(sample.shape,
    #                                                                                  dataset.feature_size))
    # data_array = np.stack(data_list)
    # # ----- ----- -----

    data_array = torch.utils.data.DataLoader(
            dataset,
            batch_size=256 * 4,
            shuffle=False,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
            )

    # ----- Apply dataset distillation -----
    seq_len = dataset.feature_size[-1] * 2
    patch_len = seq_len // 20
    ssl_kwargs = {
            'seq_len': seq_len,  # * 2 because of CRT
            'patch_len': patch_len,
            'dim': 64,
            'num_class': 2,
            'in_dim': dataset.feature_size[0],
            }
    method_kwargs = {'feature_extraction': feature_extraction, 'contamination': contamination,
                     'contamination_method': contamination_method, 'ssl_kwargs': ssl_kwargs, 'model_name_suffix': dataset_name,
                     'cache_path': cache_path, 'ssl_type': ssl_type, 'is_only_ssl': is_only_ssl, 'batch_size': 512,
                     'is_export_indices': reduction_ratio > 0.25}

    if distillation_method == "m3d" and reduction_ratio >= 0.10:
        PERCENTAGE_FOR_EACH_SPLIT = 0.05
        num_splits = int(reduction_ratio / PERCENTAGE_FOR_EACH_SPLIT)
        distilled_splits = []
        logger.warning(
                f"Split it into {num_splits} partial distillations at reduction_ratio={PERCENTAGE_FOR_EACH_SPLIT} and concatenate "
                "the results.")
        distilled_memmap = None
        start = 0
        for i in range(num_splits):
            distiller = DatasetDistiller(method="m3d", reduction_ratio=PERCENTAGE_FOR_EACH_SPLIT, **method_kwargs)
            distilled_split = distiller.distill(data_array, labels=None)
            save_path = f"{os.path.splitext(distilled_file_path)[0]}_{i}.npy"
            logger.info(f"Saving distilled split {i} to {save_path}, shape={distilled_split.shape}")
            np.save(save_path, distilled_split)
    else:
        distiller = DatasetDistiller(method=distillation_method, reduction_ratio=reduction_ratio, **method_kwargs)
        distilled_data = distiller.distill(data_array, labels=None)

        # Save distilled dataset
        if distilled_data is not None:
            np.save(distilled_file_path, distilled_data)
            logger.info(
                    f"✅ Successfully distilled the dataset from {len(dataset)} to {distilled_data.shape[0]} and saved: "
                    f"{distilled_file_path}")
    # ----- ----- -----


if __name__ == '__main__':
    fire.Fire(main)
