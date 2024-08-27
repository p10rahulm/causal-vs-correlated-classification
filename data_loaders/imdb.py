# data_loaders/imdb.py

from datasets import load_dataset
import numpy as np


def get_imdb_samples(split='train', n=None, batch_size=None):
    """
    Load IMDB dataset samples.

    Args:
    - split (str): 'train' or 'test'
    - n (int): Number of samples to return. If None, returns all samples.
    - batch_size (int): If provided, returns an iterator that yields batches of this size.

    Returns:
    - If batch_size is None: tuple (reviews, labels)
    - If batch_size is provided: iterator yielding tuples (reviews, labels) of size batch_size
    """
    dataset = load_dataset("imdb", split=split)

    if n is not None:
        dataset = dataset.select(range(min(n, len(dataset))))

    reviews = dataset['text']
    labels = dataset['label']

    if batch_size is None:
        return reviews, labels
    else:
        def batch_iterator():
            for i in range(0, len(reviews), batch_size):
                yield reviews[i:i + batch_size], labels[i:i + batch_size]

        return batch_iterator()


def get_imdb_train_samples(n=None, batch_size=None):
    """Get samples from the IMDB training set."""
    return get_imdb_samples('train', n, batch_size)


def get_imdb_test_samples(n=None, batch_size=None):
    """Get samples from the IMDB test set."""
    return get_imdb_samples('test', n, batch_size)
