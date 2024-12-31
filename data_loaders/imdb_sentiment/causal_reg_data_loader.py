#!/usr/bin/env python3
"""
causal_reg_data_loader.py

This file provides:
1) A custom Dataset that loads precomputed JSON records containing:
   - the full text (s)
   - the neutral-only text (z-only)
   - the label (0 or 1)
2) A DataModule that splits data into train/val with stratified sampling,
   and returns DataLoaders that yield (s_tokenized, z_tokenized, label).
3) An optional Test set loader (without stratification).
"""

import json
import math
import torch
from torch.utils.data import Dataset, DataLoader, Sampler, ConcatDataset, Subset
import numpy as np
from typing import List, Dict, Any, Optional


###############################################################################
# 1) A simple Dataset returning (full_text, z_text, label).
###############################################################################
class CausalRegDataset(Dataset):
    """
    Each JSON record is expected to have:
        {
          "text": <full sentence>,
          "label": <0 or 1>,
          "neutral_phrases": [ ... strings ... ],
          "causal_phrases": [ ... strings ... ]
        }
    We only need:
       - 'text' as the full sentence s
       - 'neutral_phrases' joined to form z-only
       - 'label'
    """

    def __init__(self, json_path: str):
        """
        :param json_path: Path to a JSON file containing a list of records.
        """
        super().__init__()
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        record = self.data[idx]
        full_text = record.get("text", "")
        label = record.get("label", 0)

        # Build the z-only text by joining all neutral phrases:
        neutral_phrases = record.get("neutral_phrases", [])
        z_only_text = " ".join(neutral_phrases)

        return full_text, z_only_text, label


###############################################################################
# 2) A custom Sampler for STRATIFIED BATCHING (binary classification).
###############################################################################
class StratifiedBatchSampler(Sampler[List[int]]):
    """
    Yields stratified mini-batches of indices, preserving approximate
    label proportions in each batch for binary classification.

    Steps:
      - We store the indices for label 0 and label 1 separately.
      - We shuffle each label group.
      - We chunk them out in a ratio matching the dataset proportion.

    This sampler does not do perfect partitioning across all batches,
    but ensures each batch has approx. the label distribution.

    Example usage:
       sampler = StratifiedBatchSampler(
           labels=[0, 1, 0, 1, 1, 0],
           batch_size=2,
           shuffle=True,
           drop_last=False
       )
       for batch_indices in sampler:
           ...
    """

    def __init__(
        self,
        labels: List[int],
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        generator=None,
    ):
        """
        :param labels: a list/array of 0/1 labels for each data index.
        :param batch_size: the desired batch size.
        :param shuffle: whether to shuffle the data.
        :param drop_last: if True, drop the last incomplete batch.
        :param generator: optional PyTorch RNG for reproducibility.
        """
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

        # Identify indices for label=0 and label=1
        self.indices_0 = np.where(self.labels == 0)[0]
        self.indices_1 = np.where(self.labels == 1)[0]

        # Count how many for each
        self.num_0 = len(self.indices_0)
        self.num_1 = len(self.indices_1)
        self.total_samples = self.num_0 + self.num_1

        # Compute fraction of label=1
        self.frac_1 = self.num_1 / self.total_samples
        self.frac_0 = 1.0 - self.frac_1

        # For each batch, we want (approx) batch_size * frac_0 of label=0
        # and batch_size * frac_1 of label=1 (rounded).
        # We'll sample that many from each group per batch.

        # Shuffle the indices if requested
        if self.shuffle:
            g = torch.Generator() if generator is None else generator
            seed_val = torch.randint(0, 2**32, size=(1,)).item()
            g.manual_seed(seed_val)

            self.indices_0 = self.indices_0[np.random.RandomState(seed_val).permutation(self.num_0)]
            self.indices_1 = self.indices_1[np.random.RandomState(seed_val + 1).permutation(self.num_1)]

        # We'll keep track of our position in each group
        self.pos_0 = 0
        self.pos_1 = 0

    def __iter__(self):
        """
        Yields a list of indices (one mini-batch) at a time.
        """
        while True:
            if self.pos_0 >= self.num_0 or self.pos_1 >= self.num_1:
                # we ran out of data in at least one group
                break

            # how many from 0 vs. 1 do we want?
            n_1 = int(round(self.batch_size * self.frac_1))
            n_0 = self.batch_size - n_1

            # if not enough left in group 0 or 1, then break if drop_last
            if (self.pos_0 + n_0 > self.num_0) or (self.pos_1 + n_1 > self.num_1):
                if self.drop_last:
                    break
                else:
                    # gather what remains
                    n_0 = min(n_0, self.num_0 - self.pos_0)
                    n_1 = min(n_1, self.num_1 - self.pos_1)
                    # we will still yield a smaller batch
                    # and then break

            batch_0 = self.indices_0[self.pos_0 : self.pos_0 + n_0]
            batch_1 = self.indices_1[self.pos_1 : self.pos_1 + n_1]

            self.pos_0 += n_0
            self.pos_1 += n_1

            batch_indices = np.concatenate([batch_0, batch_1])
            # If you want to shuffle the mix of 0/1 indices within the batch:
            if self.shuffle:
                np.random.shuffle(batch_indices)

            yield batch_indices.tolist()

            # if we used up everything
            if (self.pos_0 >= self.num_0) or (self.pos_1 >= self.num_1):
                break

    def __len__(self):
        """
        The number of batches we expect, ignoring drop_last vs not.
        We'll do a rough estimate (floor).
        """
        # each batch is batch_size
        # total number of possible full batches is:
        full_batches_0 = self.num_0 / (self.batch_size * self.frac_0 + 1e-8)
        full_batches_1 = self.num_1 / (self.batch_size * self.frac_1 + 1e-8)
        # approximate
        return math.floor(min(full_batches_0, full_batches_1))


###############################################################################
# 3) A DataModule that uses two CausalRegDatasets + StratifiedBatchSampler
###############################################################################
class CausalRegDataModule:
    """
    Loads two JSON files: one for train, one for val (or test).
    Creates:
      - train_dataset
      - val_dataset
    And returns DataLoaders that yield:
      {
        "full_input_ids": ...,
        "full_attention_mask": ...,
        "z_input_ids": ...,
        "z_attention_mask": ...,
        "labels": ...
      }

    The train loader uses StratifiedBatchSampler for binary classification.
    The val loader is standard (no need for stratification).
    """

    def __init__(
        self,
        train_json_path: str,
        val_json_path: str,
        tokenizer,
        batch_size: int = 32,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        """
        :param train_json_path: JSON file for training set.
        :param val_json_path:   JSON file for validation set (or test set).
        :param tokenizer:       e.g. from transformers.AutoTokenizer
        :param batch_size:      standard batch size
        :param drop_last:       whether to drop incomplete batch
        :param shuffle:         shuffle data for training
        """
        self.train_json_path = train_json_path
        self.val_json_path   = val_json_path
        self.tokenizer       = tokenizer
        self.batch_size      = batch_size
        self.drop_last       = drop_last
        self.shuffle         = shuffle

        self.train_dataset: Optional[CausalRegDataset] = None
        self.val_dataset:   Optional[CausalRegDataset] = None

        self._prepare_datasets()

    def _prepare_datasets(self):
        self.train_dataset = CausalRegDataset(self.train_json_path)
        self.val_dataset   = CausalRegDataset(self.val_json_path)

    def _collate_fn(self, batch):
        """
        We want to produce TWO sets of tokenized inputs:
          (1) full sentence
          (2) z-only sentence
        And the label.
        """
        full_texts, z_texts, labels = zip(*batch)
        # Convert to lists for tokenizer
        full_texts = list(full_texts)
        z_texts    = list(z_texts)

        # Tokenize
        full_enc = self.tokenizer(
            full_texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        z_enc = self.tokenizer(
            z_texts,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        # Convert labels to Tensor
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        # Return a dictionary that includes both versions
        return {
            "full_input_ids":      full_enc["input_ids"],
            "full_attention_mask": full_enc["attention_mask"],
            "z_input_ids":         z_enc["input_ids"],
            "z_attention_mask":    z_enc["attention_mask"],
            "labels":              labels_tensor,
        }

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader with stratified batches for train set.
        """
        # Collect all labels
        all_labels = []
        for i in range(len(self.train_dataset)):
            _, _, label = self.train_dataset[i]
            all_labels.append(label)

        # Create the stratified sampler
        sampler = StratifiedBatchSampler(
            labels=all_labels,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

        return DataLoader(
            dataset=self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self._collate_fn,
        )

    def get_val_dataloader(self) -> DataLoader:
        """
        For validation or test, we typically do a straightforward shuffle=False.
        """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            drop_last=self.drop_last,
        )

    def get_dataloaders(self):
        """
        Returns (train_loader, val_loader).
        """
        return self.get_train_dataloader(), self.get_val_dataloader()

    def get_full_train_dataloader(self) -> DataLoader:
        """
        If you want to combine train + val sets for final training, do it here.
        We'll still do stratified sampling across the combined data.
        """
        # Merge both datasets
        merged_dataset = ConcatDataset([self.train_dataset, self.val_dataset])

        # Collect all labels from the combined data
        # But we need to index carefully: first len(train_dataset) are train, etc.
        all_labels = []
        len_train = len(self.train_dataset)
        for i in range(len_train):
            _, _, label = self.train_dataset[i]
            all_labels.append(label)
        for j in range(len(self.val_dataset)):
            _, _, label = self.val_dataset[j]
            all_labels.append(label)

        sampler = StratifiedBatchSampler(
            labels=all_labels,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            drop_last=self.drop_last,
        )

        return DataLoader(
            dataset=merged_dataset,
            batch_sampler=sampler,
            collate_fn=self._collate_fn,
        )
