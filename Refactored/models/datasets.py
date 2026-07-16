import logging
from torch.utils.data import Dataset
from typing import List

logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    """Wraps text data for efficient batched loading in PyTorch."""
    def __init__(self, texts: List[str]):
        self.texts = texts
        logger.debug(f"Initialized TextDataset with {len(texts)} records.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return str(self.texts[idx])