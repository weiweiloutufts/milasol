import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
import random
from pathlib import Path

# Define amino acid mapping
aa_vocab = {
    "A": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "K": 9,
    "L": 10,
    "M": 11,
    "N": 12,
    "P": 13,
    "Q": 14,
    "R": 15,
    "S": 16,
    "T": 17,
    "V": 18,
    "W": 19,
    "Y": 20,
    "-": 0,
    "X": 21,
}


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def random_deletion(seq, p=0.05):
    return "".join([aa for aa in seq if random.random() > p])


def random_substitution(seq, p=0.05):
    return "".join(
        [aa if random.random() > p else random.choice(AMINO_ACIDS) for aa in seq]
    )


def random_masking(seq, p=0.05, mask_token="X"):
    return "".join([aa if random.random() > p else mask_token for aa in seq])


def _is_pathlike(obj):
    return isinstance(obj, (str, Path))


def _load_table_as_tensor(source, *, dtype, header=None, name="input"):
    if source is None:
        return None
    if isinstance(source, torch.Tensor):
        tensor = source.detach().clone()
        if dtype is not None and tensor.dtype != dtype:
            tensor = tensor.to(dtype=dtype)
        return tensor
    if isinstance(source, (list, tuple)):
        return torch.tensor(source, dtype=dtype)
    if isinstance(source, pd.DataFrame):
        return torch.tensor(source.values, dtype=dtype)
    if _is_pathlike(source):
        frame = pd.read_csv(source, header=header)

        return torch.tensor(frame.values, dtype=dtype)
    raise TypeError(
        f"{name} must be a path, torch.Tensor, pandas DataFrame, or a sequence of values."
    )


def _prepare_sequence_source(source):
    if source is None:
        raise ValueError(
            "A sequence source must be provided as a path or tensor-like input."
        )
    if isinstance(source, torch.Tensor):
        tensor = source.detach().clone().to(dtype=torch.long)
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return None, tensor
    if isinstance(source, (list, tuple)):
        if len(source) == 0:
            raise ValueError("Sequence input cannot be empty.")
        if all(isinstance(item, str) for item in source):
            return [str(item) for item in source], None
        if all(torch.is_tensor(item) for item in source):
            tensors = [item.detach().clone().to(dtype=torch.long) for item in source]
            return None, tensors
        raise TypeError(
            "Sequence inputs must be all strings or all torch.Tensor objects."
        )
    if isinstance(source, pd.Series):
        return source.astype(str).tolist(), None
    if isinstance(source, pd.DataFrame):
        return source.iloc[:, 0].astype(str).tolist(), None
    if _is_pathlike(source):
        frame = pd.read_csv(source, header=None)
        return frame.iloc[:, 0].astype(str).tolist(), None
    raise TypeError(
        "sequence_file must be a filepath, tensor, pandas object, or sequence of strings/tensors."
    )


def collate_fn(batch):
    sequences, esm_embs, prot_embs, raygun_embs, solu_feats, labels, idx = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    max_len = 1241
    if max_len is not None:
        if sequences_padded.size(1) < max_len:
            pad_size = max_len - sequences_padded.size(1)
            padding = torch.full(
                (sequences_padded.size(0), pad_size), 0, dtype=sequences_padded.dtype
            )
            sequences_padded = torch.cat([sequences_padded, padding], dim=1)
        elif sequences_padded.size(1) > max_len:
            sequences_padded = sequences_padded[:, :max_len]
    esm_embs = torch.stack(esm_embs)
    prot_embs = torch.stack(prot_embs)
    raygun_embs = torch.stack(raygun_embs)
    solu_feats = torch.stack(solu_feats)
    labels = torch.stack(labels)
    idx = torch.as_tensor(idx, dtype=torch.long)
    return sequences_padded, esm_embs, prot_embs, raygun_embs, solu_feats, labels, idx


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sequence_file,
        esm_file,
        prot_file,
        raygun_file,
        feats_file=None,
        label_file=None,
        augment=False,
    ):
        self.sequence_strings, self.sequence_tensors = _prepare_sequence_source(
            sequence_file
        )

        self.df_sequences = (
            pd.DataFrame(self.sequence_strings, columns=[0])
            if self.sequence_strings is not None
            else None
        )

        if self.sequence_strings is not None:
            self._num_samples = len(self.sequence_strings)
        elif isinstance(self.sequence_tensors, torch.Tensor):
            self._num_samples = self.sequence_tensors.size(0)
        else:
            self._num_samples = len(self.sequence_tensors)

        self.esm_embeddings = _load_table_as_tensor(
            esm_file, dtype=torch.float, header=0, name="esm_file"
        )
        if self.esm_embeddings.size(0) != self._num_samples:
            print(self.esm_embeddings.size(0), self._num_samples)
            raise ValueError("Mismatch between ESM embeddings  and sequence inputs.")

        self.prot_embeddings = _load_table_as_tensor(
            prot_file, dtype=torch.float, header=0, name="prot_file"
        )
        if self.prot_embeddings.size(0) != self._num_samples:
            raise ValueError("Mismatch between Prot embeddings and sequence inputs.")

        self.raygun_embeddings = _load_table_as_tensor(
            raygun_file, dtype=torch.float, header=0, name="raygun_file"
        )
        if self.raygun_embeddings.size(0) != self._num_samples:
            raise ValueError("Mismatch between RayGun embeddings and sequence inputs.")

        if feats_file is not None:
            self.solu_feats = _load_table_as_tensor(
                feats_file, dtype=torch.float, name="feats_file"
            )
            if self.solu_feats.size(0) != self._num_samples:
                raise ValueError("Mismatch between Solu features and other inputs.")
        else:
            self.solu_feats = None

        if label_file is not None:
            self.labels = _load_table_as_tensor(
                label_file,
                dtype=torch.float,
                header=None,
                name="label_file",
            )
            if self.labels.size(0) != self._num_samples:
                raise ValueError(
                    "Mismatch in number of samples between labels and inputs."
                )
        else:
            self.labels = None

        if augment and self.sequence_strings is None:
            raise ValueError(
                "Augmentation requires raw sequence strings; provide sequence strings instead of tensors."
            )

        self.augment = augment

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        if self.sequence_strings is not None:
            sequence_str = self.sequence_strings[idx]
            seq = sequence_str
            if self.augment:
                # Apply augmentations randomly when raw strings are available
                if random.random() < 0.5:
                    seq = random_deletion(sequence_str)
                if random.random() < 0.5:
                    seq = random_substitution(sequence_str)
                if random.random() < 0.5:
                    seq = random_masking(sequence_str)
            sequence_tensor = torch.tensor(
                [aa_vocab.get(aa, 21) for aa in seq], dtype=torch.long
            )
        else:

            base_tensor = self.sequence_tensors[idx]
            sequence_tensor = base_tensor.detach().clone().to(dtype=torch.long)

        esm_output = self.esm_embeddings[idx]
        prot_output = self.prot_embeddings[idx]
        raygun_output = self.raygun_embeddings[idx]
        if self.labels is not None and self.solu_feats is not None:
            # Training with labels + extra features
            label = self.labels[idx]
            solu_output = self.solu_feats[idx]
            return (
                sequence_tensor,
                esm_output,
                prot_output,
                raygun_output,
                solu_output,
                label,
                idx,
            )

        elif self.labels is not None:
            # Training with labels, but no solubility features
            label = self.labels[idx]
            return (
                sequence_tensor,
                esm_output,
                prot_output,
                raygun_output,
                torch.tensor(-1),
                label,
                idx,
            )

        else:
            # Inference: no labels
            return (
                sequence_tensor,
                esm_output,
                prot_output,
                raygun_output,
                torch.tensor(-1),
                torch.tensor(-1),
                idx,
            )
