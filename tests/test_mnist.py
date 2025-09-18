import datasets
import pytest

from continual_learning.configs.dataset import DatasetConfig
from continual_learning.data.mnist import ProcessMNIST, SplitMNIST


@pytest.fixture
def dataset():
    return datasets.load_dataset("mnist").with_format("numpy")["train"]  # pyright: ignore[reportIndexIssue]


def test_process_mnist(dataset):
    test_element = dataset[0]
    processed_element = ProcessMNIST(flatten=False).map(test_element)
    assert isinstance(processed_element, tuple)
    assert len(processed_element) == 2
    x, y = processed_element
    assert x.shape == (28, 28)
    assert y.shape == (10,)
    assert y.argmax().item() == test_element["label"]


def test_split_mnist():
    num_tasks = 5
    batch_size = 32
    ds = SplitMNIST(
        DatasetConfig(
            name="split-mnist",
            num_tasks=num_tasks,
            num_epochs_per_task=2,
            seed=42,
            batch_size=batch_size,
        )
    )

    for task in ds.tasks:
        train, test = task
        for batch in train:
            x, y = batch
            assert len(x) == batch_size
            assert len(y) == batch_size
            break
        for batch in test:
            x, y = batch
            assert len(x) == batch_size
            assert len(y) == batch_size
            break
