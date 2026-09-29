"""Fixed channel subsets for the 14/8/4-channel benchmark (config.CHANNEL_SUBSETS)."""
from .config import CHANNEL_SUBSETS, EEG_NAMES


def subset_indices(n_channels):
    if n_channels not in CHANNEL_SUBSETS:
        raise ValueError(f"n_channels must be one of {list(CHANNEL_SUBSETS)}")
    names = CHANNEL_SUBSETS[n_channels]
    return [EEG_NAMES.index(n) for n in names], names


def take(eeg, n_channels):
    idx, names = subset_indices(n_channels)
    return eeg[:, idx], names