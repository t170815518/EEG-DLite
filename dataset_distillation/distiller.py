# dataset_distillation/distiller.py
from loguru import logger
from typing_extensions import Literal
try:
    from .methods import *
except ImportError as e:
    logger.warning(e)
    from methods import *


class DatasetDistiller:
    def __init__(self, method: Literal['random', 'm3d', 'coreset'] = "random", reduction_ratio=2, function=None,
                 **kwargs):
        """
        Initializes the dataset distiller.

        :param method: str, distillation type ("random", "function_based")
        :param reduction_ratio: int, factor by which to reduce dataset
        :param function: callable, custom function for filtering if function-based
        """
        self.method = method.lower()
        self.reduction_ratio = reduction_ratio
        self.function = function
        self.kwargs = kwargs

    def distill(self, dataset, labels):
        """
        Applies the chosen distillation method.

        :param dataset: list or np.array, feature dataset
        :param labels: list or np.array, corresponding labels
        :return: tuple (distilled_dataset, distilled_labels)
        """
        if self.method == "random":
            return RandomDistillation().apply(dataset, labels, self.reduction_ratio, **self.kwargs)
        elif self.method == "m3d":
            return (M3DSyntheticDistillation(dataset, labels, reduction_ratio=self.reduction_ratio, **self.kwargs).
                    apply(dataset, labels))
        elif self.method == 'coreset':
            return KCenterDistillation(self.reduction_ratio, **self.kwargs).apply(dataset, labels)
        else:
            raise ValueError(f"Unknown distillation method: {self.method}")

    def __repr__(self):
        """
        useful for logging.
        """
        return (f"DatasetDistiller(method='{self.method}', "
                f"reduction_ratio={self.reduction_ratio}, "
                f"kwargs={self.kwargs})")
