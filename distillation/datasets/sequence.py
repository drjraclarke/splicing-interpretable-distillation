import numpy as np
import pandas as pd
import torch
from typing import List, Optional

nucleotide_map = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    # All-zero encoding for padding inserted when pooling variable-length
    # designs into a common fixed-length input.
    "N": [0, 0, 0, 0],
}

# Lookup table from ASCII byte value -> base index (0-3), -1 for anything
# else (e.g. 'N'), which one_hot_encode_sequences_vectorized turns into an
# all-zero one-hot vector - matches nucleotide_map above.
_BASE_LUT = np.full(256, -1, dtype=np.int8)
for _base, _vec in nucleotide_map.items():
    if any(_vec):
        _BASE_LUT[ord(_base)] = _vec.index(1)


def one_hot_encode_sequence(sequence: str) -> List[List[int]]:
    return [nucleotide_map[nucleotide] for nucleotide in sequence]


def one_hot_encode_sequences_vectorized(sequences: List[str]) -> np.ndarray:
    """
    Vectorized one-hot encoding of many equal-length sequences at once.

    The original per-sequence Python-list approach (one_hot_encode_sequence,
    applied via pandas .apply then torch.tensor(nested_list)) allocates a
    Python object (int/list) per base, which becomes prohibitive at the
    scale of a multi-design pooled dataset (millions of rows): e.g. even
    34,741 sequences of ~535nt needed 24GB+ RAM this way. This numpy version
    encodes all sequences in one pass with no per-base Python objects -
    equivalent float32 output, ~50-100x less peak memory.

    Returns an array of shape (n_sequences, seq_len, 4).
    """
    seq_len = len(sequences[0])
    assert all(len(s) == seq_len for s in sequences), "all sequences must be the same length"

    joined = "".join(sequences).encode("ascii")
    byte_arr = np.frombuffer(joined, dtype=np.uint8).reshape(len(sequences), seq_len)
    base_idx = _BASE_LUT[byte_arr]  # (n, seq_len), values in {-1, 0, 1, 2, 3}

    one_hot = np.zeros((len(sequences), seq_len, 4), dtype=np.float32)
    for base in range(4):
        one_hot[:, :, base] = base_idx == base
    return one_hot


class SequenceDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for DNA sequences.
    """

    def __init__(
        self,
        data_path: Optional[str],
        data_df: Optional[pd.DataFrame],
        sequence_column: str,
        target_column: str,
        upstream_sequence: str = "",
        downstream_sequence: str = "",
    ):
        # Ensure that only one of data_path and data_df is provided
        assert data_path is not None or data_df is not None
        assert data_path is None or data_df is None

        if data_path is not None:
            self.data = pd.read_csv(data_path)
        else:
            self.data = data_df

        full_sequences = (
            self.data[sequence_column]
            .apply(lambda x: f"{upstream_sequence}{x}{downstream_sequence}")
            .tolist()
        )
        one_hot = one_hot_encode_sequences_vectorized(full_sequences)
        self.sequences = torch.from_numpy(one_hot).permute(0, 2, 1)

        self.labels = torch.tensor(
            self.data[target_column].to_list(), dtype=torch.float32
        )

        # Ensure that the number of sequences and labels match
        assert len(self.sequences) == len(self.labels)

        # Ensue that sequences have correct channel sizes
        assert self.sequences.size(1) == 4

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]
