import os.path
from loguru import logger
import numpy as np
import h5py
import bisect
from pathlib import Path
from typing import List
from torch.utils.data import Dataset

list_path = List[Path]


class SingleShockDataset(Dataset):
    """Read single hdf5 file regardless of label, subject, and paradigm."""

    def __init__(self, file_path: Path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1, sample_percentage=0.5, is_return_subject_id: bool = False):
        """
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride_size: the interval between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file
            (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        """
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.sample_ratio = sample_percentage
        self.is_return_subject_id = is_return_subject_id

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__subjects2sampled_idxes = {}
        self.__global_idxes = []
        self.__local_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        for subject_id, subject in enumerate(self.__subjects):
            self.__global_idxes.append(global_idx)  # the start index of the subject's sample in the dataset
            subject_len = self.__file[subject]['eeg'].shape[1]
            # total number of samples
            total_sample_num = (subject_len - self.__window_size) // self.__stride_size + 1
            # cut out part of samples
            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size
            self.__local_idxes.append(start_idx)
            global_idx_ = (end_idx - start_idx) // self.__stride_size + 1
            global_idx += global_idx_
            sampled_idxes = np.arange(global_idx_ - 1)

            abnormal_count = 0
            if len(sampled_idxes) > 0:
                sampled_idxes = np.random.choice(sampled_idxes, size=int(len(sampled_idxes) * self.sample_ratio),
                                                 replace=False)
                sampled_idxes = np.sort(sampled_idxes)
            else:
                logger.debug('len(sampled_idxes) > 0 for subject {} in {}'.format(subject, self.__file))
                sampled_idxes = []
                abnormal_count += 1
            self.__subjects2sampled_idxes[subject_id] = sampled_idxes
        self.__length = int(global_idx * self.sample_ratio)
        if abnormal_count > 0:
            logger.warning(f'{abnormal_count} subjects with empty sampled_idxes')

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
        idx_ = idx - self.__global_idxes[subject_idx]
        idx_ = bisect.bisect(self.__subjects2sampled_idxes[subject_idx], idx_)
        item_start_idx = idx_ * self.__stride_size + self.__local_idxes[subject_idx]
        eeg_segment = self.__file[self.__subjects[subject_idx]]['eeg'][:,
                      item_start_idx:item_start_idx + self.__window_size]
        if self.is_return_subject_id:
            return eeg_segment, subject_idx
        else:
            return eeg_segment

    def free(self) -> None:
        if self.__file:
            self.__file.close()
            self.__file = None

    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']

    def get_filepath(self):
        return self.__file_path


class DistillatedSingleShockDataset(SingleShockDataset):
    """
    The distillated files are stored in .npy format
    """

    def __init__(self, file_path: Path, distillated_path: Path, window_size: int = 200, stride_size: int = 1,
                 start_percentage: float = 0,
                 end_percentage: float = 1, sample_percentage=0.5, ):
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.sample_ratio = sample_percentage

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__subjects2sampled_idxes = {}
        self.__global_idxes = []
        self.__local_idxes = []

        self.distillated_path = distillated_path
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]
        self.data = np.load(os.path.join(self.distillated_path, self.__file_path.stem + '.npy'),
                            mmap_mode='r')

        self.__length = len(self.data)

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    def __getitem__(self, idx: int):
        return self.data[idx]

    def __len__(self):
        return self.__length

    @property
    def feature_size(self):
        return self.__feature_size

    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


class ShockDataset(Dataset):
    """integrate multiple hdf5 files or .npy files"""

    def __init__(self, file_paths: list_path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1, sample_percentage: float = 1.0, distillated_path: Path = None,
                 is_return_subject_id: bool = False):
        """
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        """
        self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.__sample_percentage = sample_percentage
        self.__distillated_path = distillated_path
        self.is_return_subject_id = is_return_subject_id

        self.__datasets = []
        self.__length = None
        self.__feature_size = None

        self.__dataset_idxes = []

        self.__init_dataset()

    def __init_dataset(self) -> None:
        if self.__distillated_path is None:  # load .hd5f files
            self.__datasets = [
                SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage,
                                   self.__end_percentage, self.__sample_percentage, self.is_return_subject_id) for file_path in self.__file_paths]
        else:  # load .npy files
            for file_path in self.__file_paths:
                try:
                    self.__datasets.append(
                        DistillatedSingleShockDataset(file_path, self.__distillated_path, self.__window_size,
                                                      self.__stride_size, self.__start_percentage,
                                                      self.__end_percentage, self.__sample_percentage))
                except FileNotFoundError:
                    logger.warning('File {} is skipped due to FileNotFoundError'.format(file_path))
                    continue
        # calculate the number of samples for each subdataset to form the integral indexes
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    @property
    def file_paths(self):
        return self.__file_paths

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]

    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()

    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()

    def get_datasets(self):
        return self.__datasets


class DownstreamDataset(Dataset):
    """读取单个hdf5文件，仅使用范式中途采集的数据，标签内包含范式标签与被试性别，如有需要可以继续往字典中添加"""

    def __init__(self, file_path: Path, window_size: int = 200, stride_size: int = 1, start_percentage: float = 0,
                 end_percentage: float = 1, trial_start_percentage: float = 0, trial_end_percentage: float = 1,
                 subject_start_percentage: float = 0, subject_end_percentage: float = 1):
        '''
        从路径file_path中提取数据集。

        :param Path file_path: 目标数据路径
        :param int window_size: 单个样本长度
        :param int stride_size: 两个相邻样本间隔
        :param float start_percentage: 数据集中，每个采纳的trial内首个样本在此trial的样本中的百分比索引（包括）。
        :param float end_percentage: 数据集中，每个采纳的trial内末尾样本在此trial的样本中的百分比索引（不包括）。
        :param float trial_start_percentage: 数据集中，采纳的首个trial在此被试的所有trial中的百分比索引（包括）。
        :param float trial_end_percentage: 数据集中，采纳的末个trial在此被试的所有trial中的百分比索引（不包括）。
        :param float subject_start_percentage: 数据集中，采纳的首个被试的百分比索引（包括）。
        :param float subject_end_percentage: 数据集中，采纳的末个被试的百分比索引（不包括）。

        比如，数据文件总共10个被试，每个被试有15个trial，每个trial提供100个样本时。取参数为0.2, 0.8, 0.34, 0.67, 0.2, 0.8时，数据集会包括下标为[2, 8)的被试，每个被试的下标为[5, 10)的trial中，每个trial下标为[20, 80)的样本。
        '''
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.__trial_start_percentage = trial_start_percentage
        self.__trial_end_percentage = trial_end_percentage
        self.__subject_start_percentage = subject_start_percentage
        self.__subject_end_percentage = subject_end_percentage

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = []  # 从第几个样本开始是哪个被试
        self.__local_idxess = []  # 从这个被试的第几个样本开始是哪个trial
        self.__trial_start_idxess = []  # trial开始索引
        self.__genders = []
        self.__labelss = []

        self.__rsFreq = None

        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        subject_start_id = int(len(self.__subjects) * self.__subject_start_percentage)  # 包括在数据集中的被试开始id
        subject_end_id = int(len(self.__subjects) * self.__subject_end_percentage - 1)  # 包括在数据集中的被试结束id
        for subject_id, subject in enumerate(self.__subjects):
            self.__global_idxes.append(global_idx)
            self.__genders.append(self.__file[subject].attrs['gender'])
            self.__labelss.append(self.__file[subject].attrs['label'])
            self.__rsFreq = self.__file[subject]['eeg'].attrs['rsFreq']

            local_idxes = []  # 当前trial的第一个样本在数据集中的样本索引
            trial_start_idxes = []  # 当前trial在原始数据中的开始位置索引
            trial_starts = self.__file[subject].attrs['trialStart']
            trial_ends = self.__file[subject].attrs['trialEnd']
            local_idx = 0
            if subject_id >= subject_start_id and subject_id <= subject_end_id:
                trial_start_id = int(len(trial_starts) * self.__trial_start_percentage)  # 该被试包括在数据集中的trial开始id
                trial_end_id = int(len(trial_starts) * self.__trial_end_percentage - 1)  # 该被试包括在数据集中的trial结束id
                for trial_id, (trial_start, trial_end) in enumerate(zip(trial_starts, trial_ends)):
                    local_idxes.append(local_idx)

                    if trial_id >= trial_start_id and trial_id <= trial_end_id:
                        trial_len = (trial_end - trial_start + 1) * self.__rsFreq
                        trial_sample_num = (trial_len - self.__window_size) // self.__stride_size + 1
                        start_idx = int(
                            trial_sample_num * self.__start_percentage) * self.__stride_size + trial_start * self.__rsFreq
                        end_idx = int(
                            trial_sample_num * self.__end_percentage - 1) * self.__stride_size + trial_start * self.__rsFreq

                        trial_start_idxes.append(start_idx)
                        local_idx += (end_idx - start_idx) // self.__stride_size + 1
                    else:
                        trial_start_idxes.append(0)

            self.__local_idxess.append(local_idxes)
            self.__trial_start_idxess.append(trial_start_idxes)

            global_idx += local_idx

        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    @property
    def rsfreq(self):
        return self.__rsFreq

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        # 先确认样本属于哪个被试，再确认样本属于哪个trial
        subject_id = bisect.bisect(self.__global_idxes, idx) - 1
        trial_id = bisect.bisect(self.__local_idxess[subject_id], idx - self.__global_idxes[subject_id]) - 1
        item_start_idx = (idx - self.__global_idxes[subject_id] - self.__local_idxess[subject_id][
            trial_id]) * self.__stride_size + self.__trial_start_idxess[subject_id][trial_id]

        labels = {}
        labels['gender'] = self.__genders[subject_id]
        labels['label'] = self.__labelss[subject_id][trial_id]

        return self.__file[self.__subjects[subject_id]]['eeg'][:,
               item_start_idx:item_start_idx + self.__window_size], labels

    def free(self) -> None:
        # TODO 临时方案，目标：减少文件打开次数。查一下flush
        if self.__file:
            self.__file.close()
            self.__file = None

    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


if __name__ == '__main__':
    # Define the path to your HDF5 file
    # h5_file_path = Path(
    #     "/home/neurips2025/dataset_maker/h5Data/raweegdata.hdf5")  # Replace with the actual file path
    # # ----- Test single dataset -----
    # # Define the path to your HDF5 file
    # h5_file_path = Path("/home/yuting/data/h5data/bcicompetitioniv1.hdf5")  # Replace with the actual file path
    #
    # # Initialize the dataset
    # dataset = SingleShockDataset(h5_file_path, window_size=200, stride_size=400, )
    #
    # # Print the number of samples in the dataset
    # print(f"Total number of samples: {len(dataset)}")
    #
    # # Print feature size (EEG data structure)
    # print(f"Feature size: {dataset.feature_size}")
    #
    # # Print channel names (if available)
    # try:
    #     ch_names = dataset.get_ch_names()
    #     print(f"Channel names: {ch_names}")
    # except AttributeError:
    #     print("Channel names not found in the dataset.")
    # # ----- ----- -----

    # ----- iterate over the dataset with dataloader -----
    import torch
    from torch.utils.data import DataLoader

    # Initialize the dataset
    dataset = SingleShockDataset(h5_file_path, window_size=200, stride_size=400, )
    # Define dataset paths
    h5_paths = [
        Path("/home/yuting/data/h5data/bcicompetitioniv1.hdf5"),
    ]

    # Print the number of samples in the dataset
    print(f"Total number of samples: {len(dataset)}")
    # Create dataset instance with random sampling (e.g., 50% of data)
    dataset = ShockDataset(h5_paths, window_size=200, stride_size=200, sample_percentage=0.5)

    # Print feature size (EEG data structure)
    print(f"Feature size: {dataset.feature_size}")
    # Define DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)

    # Print channel names (if available)
    try:
        ch_names = dataset.get_ch_names()
        print(f"Channel names: {ch_names}")
    except AttributeError:
        print("Channel names not found in the dataset.")
    # Iterate over the dataset
    print(f"Total Samples After Sampling: {len(dataset)}")

    for batch_idx, batch in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}: {batch.shape}")
        if batch_idx == 2:  # Limit to 3 batches for quick testing
            break

    print("Test iteration completed successfully.")
    # ----- ----- -----
