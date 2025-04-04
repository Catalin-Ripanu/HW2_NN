"""
This module contains code for working with datasets and dataloader.
"""

from typing import Any, Callable
import pandas as pd
import torch
from torch.utils import data
import constants, tokenization


class Multi30kDataset(data.Dataset):
    """Bilingual dataset which deals with converting tokenized text into indices."""

    def __init__(
        self,
        df: pd.DataFrame,
        vocab_en: tokenization.Vocab,
        vocab_fr: tokenization.Vocab,
        max_len: int = None,  # Optional maximum sequence length
        min_len: int = 2  # Minimum sequence length (SOS + EOS)
    ) -> None:
        super().__init__()

        # Pre-calculate common values
        self.en = []
        self.fr = []
        unknown_en = vocab_en.token_to_index(constants.UNKNOWN)
        unknown_fr = vocab_fr.token_to_index(constants.UNKNOWN)
        sos_en = vocab_en.token_to_index(constants.SOS)
        eos_en = vocab_en.token_to_index(constants.EOS)
        sos_fr = vocab_fr.token_to_index(constants.SOS)
        eos_fr = vocab_fr.token_to_index(constants.EOS)
        
        # Pre-allocate lists
        self.en = [None] * len(df)
        self.fr = [None] * len(df)
        valid_indices = []

        def to_indices(words: list[str], vocab: tokenization.Vocab, unknown_idx: int,
                      sos_idx: int, eos_idx: int) -> list[int]:
            available_len = max_len - 2 if max_len else len(words)
            indices = [sos_idx]
            indices.extend(vocab.token_to_index(w) if vocab.token_to_index(w) < len(vocab) 
                         else unknown_idx for w in words[:available_len])
            indices.append(eos_idx)
            return indices

        # Process the data
        skipped = 0
        
        for idx, row in df.iterrows():
            try:
                en_indices = to_indices(list(row["en"]), vocab_en, unknown_en, sos_en, eos_en)
                fr_indices = to_indices(list(row["fr"]), vocab_fr, unknown_fr, sos_fr, eos_fr)
                
                # Validate sequence lengths
                if len(en_indices) < min_len or len(fr_indices) < min_len:
                    print(f"Warning: Row {idx} has sequence shorter than minimum length")
                    skipped += 1
                    continue
                    
                if max_len and (len(en_indices) > max_len or len(fr_indices) > max_len):
                    print(f"Warning: Row {idx} exceeds maximum length")
                    skipped += 1
                    continue
                
                self.en[idx] = en_indices
                self.fr[idx] = fr_indices
                
            except Exception as e:
                print(f"Error processing row {idx}: {str(e)}")
                skipped += 1
                
        if skipped > 0:
            print(f"Skipped {skipped} rows due to invalid indices or length")

    def __len__(self) -> int:
        return len(self.en)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        return self.en[index], self.fr[index]

def make_collator(
    vocab_en: tokenization.Vocab,
    vocab_fr: tokenization.Vocab,
) -> Callable[[list[Any]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Builds a collator function that ensures all indices are within vocabulary bounds
    and properly handles sequence lengths.
    """
    
    # Pre-calculate indices
    en_pad_index = vocab_en.token_to_index(constants.PAD)
    fr_pad_index = vocab_fr.token_to_index(constants.PAD)
    unknown_en = vocab_en.token_to_index(constants.UNKNOWN)
    unknown_fr = vocab_fr.token_to_index(constants.UNKNOWN)
    
    def fn(samples: list[Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Convert to tensors first
        en_seqs = [torch.tensor(en, dtype=torch.long) for en, _ in samples]
        fr_seqs = [torch.tensor(fr, dtype=torch.long) for _, fr in samples]
        
        # Get lengths and pad in one go
        lengths_en = torch.tensor([len(en) for en in en_seqs])
        
        # Pad sequences using torch operations
        padded_en = torch.nn.utils.rnn.pad_sequence(
            en_seqs, batch_first=True, padding_value=en_pad_index)
        padded_fr = torch.nn.utils.rnn.pad_sequence(
            fr_seqs, batch_first=True, padding_value=fr_pad_index)
        
        # Replace out-of-vocab indices
        padded_en = torch.where(padded_en < len(vocab_en), padded_en, unknown_en)
        padded_fr = torch.where(padded_fr < len(vocab_fr), padded_fr, unknown_fr)
        
        return padded_fr, lengths_en, padded_en
    
    return fn
