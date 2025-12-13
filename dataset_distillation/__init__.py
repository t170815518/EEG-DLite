# dataset_distillation/__init__.py
# Initializes the dataset distillation package
import numpy as np
from tqdm import tqdm
from loguru import logger

try:
    from .distiller import DatasetDistiller
    from .methods import RandomDistillation
except ImportError as e:
    logger.warning(e)
    from distiller import DatasetDistiller
    from methods import RandomDistillation


def extract_feature_matrix():
    global dataset, features, labels, data
    import torcheeg.transforms as transforms
    from torcheeg.models import EEGNet
    from torcheeg.datasets import SEEDDataset
    from torcheeg.model_selection import KFoldGroupbyTrial

    dataset = SEEDDataset(root_path=r'F:\Preprocessed_EEG\Preprocessed_EEG',
                          io_path=r'C:\Users\CBCR-C\Documents\GitHub\Large-Brain-Model\preliminary_experiment\.torcheeg\datasets_1738644318567_NfikJ',
                          online_transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.To2d()
                                  ]),
                          label_transform=transforms.Compose([
                                  transforms.Select('emotion'),
                                  transforms.Lambda(lambda x: x + 1)
                                  ]),
                          num_worker=32)
    # Extract features and labels
    features = []
    labels = []
    for data in dataset:
        data, label = data
        features.append(data.numpy())  # Flattening ensures a 1D vector
        labels.append(label)
    # Convert to NumPy arrays
    features_matrix = np.array(features)  # Shape: (num_samples, num_features)
    labels_array = np.array(labels)  # Shape: (num_samples, )
    print("Feature matrix shape:", features_matrix.shape)
    print("Labels shape:", labels_array.shape)
    np.save('features.npy', features)
    np.save('labels.npy', labels)


def extract_feature_matrix_mobi():
    global dataset, features, labels, data
    import torcheeg.transforms as transforms
    from torcheeg.models import EEGNet
    from torcheeg.datasets import MoBIDataset
    from torcheeg.model_selection import KFoldGroupbyTrial
    dataset = MoBIDataset(root_path="/home/yuting/Large-Brain-Model/MoBI/", chunk_size=200, offset=0, overlap=195,
                          num_channel=64, online_transform=transforms.Compose([transforms.ToTensor(),
                                                                               transforms.To2d(), ]),
                          offline_transform=transforms.Compose([  # fixme: implement the transform
                                  transforms.BandSignal(100, band_dict={'default': [0.1, 49]}),
                                  ]),
                          label_transform=transforms.Lambda(lambda y: y['label'] / 90, targets=['y']),
                          io_mode='pickle',
                          io_path='/home/yuting/Large-Brain-Model/preliminary_experiment/.torcheeg/datasets_1740208325851_nM1QM/',
                          num_worker=32)
    # Extract features and labels
    logger.info('Reading features and labels from dataset')
    features = []
    labels = []
    for data in tqdm(dataset, desc='Reading features and labels from training dataset'):
        data, label = data
        features.append(data.numpy())  # Flattening ensures a 1D vector
        labels.append(label)
    features = np.array(features)  # Shape: (num_samples, num_features)
    labels = np.array(labels)  # Shape: (num_samples, )
    logger.info("Feature matrix shape: {}\tLabels shape: {}".format(features.shape, labels.shape))
    np.save('features.npy', features)
    np.save('labels.npy', labels)


if __name__ == '__main__':
    import numpy as np
    import time
    from loguru import logger

    # Set up the log file
    logfile_name = 'log/distill-{}.log'.format(int(time.time()))
    logger.add(logfile_name, level="DEBUG")

    # extract_feature_matrix()

    # extract_feature_matrix_mobi()

    ##### Unit Test of Random Sampling (MoBI) #####
    features = np.load('features.npy')
    labels = np.load('labels.npy')
    # classes, sizes = np.unique(labels, return_counts=True)
    # class2initial_size = {k: v for (k, v) in zip(classes, sizes)}
    distiller_kwargs = {}
    distiller_kwargs['is_sample_by_class'] = False
    distiller = DatasetDistiller(method="random", reduction_ratio=0.05, is_debug=False, loss_type='mse',
                                 feature_extraction='SSL', **distiller_kwargs)
    distilled_data, distilled_labels = distiller.distill(features, labels)

    # ##### Unit Test of Coreset Construction #####
    # features = np.load('features.npy')
    # labels = np.load('labels.npy')
    # classes, sizes = np.unique(labels, return_counts=True)
    # class2initial_size = {k: v for (k, v) in zip(classes, sizes)}
    #
    # distiller = DatasetDistiller(method="coreset", reduction_ratio=0.05, class2initial_size=class2initial_size,
    #                              features=features, labels=labels, is_debug=False, loss_type='mse',
    #                              feature_extraction='SSL')
    # distilled_data, distilled_labels = distiller.distill(features, labels)
    ###############

    # ##### Unit Test of M3D #####
    # features = np.load('features.npy')
    # labels = np.load('labels.npy')
    # classes, sizes = np.unique(labels, return_counts=True)
    # class2initial_size = {k: v for (k, v) in zip(classes, sizes)}
    #
    # distiller = DatasetDistiller(method="M3D", reduction_ratio=0.01, class2initial_size=class2initial_size,
    #                              features=features, labels=labels, is_debug=False, loss_type='mse')
    # distilled_data, distilled_labels = distiller.distill(features, labels)
    # ###############
