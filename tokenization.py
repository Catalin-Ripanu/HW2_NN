"""
This module contains code which helps with tokenizing text.
"""

from typing import Iterable
import constants
from collections import Counter


class Vocab:
    """A fixed vocabulary which supports unknown words."""
    def __init__(self, tokens: set[str], sos: str, eos: str, pad: str, unknown_token: str) -> None:
        # First add special tokens in a fixed order
        special_tokens = [pad, sos, eos, unknown_token]
        for token in special_tokens:
            if token in tokens:
                tokens.remove(token)
       
        # Now create the final token list with special tokens first
        self.tokens = special_tokens + sorted(tokens)
        self.indices = {token: index for index, token in enumerate(self.tokens)}
        
        # Store special token info
        self.pad_token = pad
        self.sos_token = sos
        self.eos_token = eos
        self.unknown_token = unknown_token
        
        self.pad_index = self.indices[pad]
        self.sos_index = self.indices[sos]
        self.eos_index = self.indices[eos]
        self.unknown_index = self.indices[unknown_token]
       
        # Add validation
        assert self.tokens[self.pad_index] == pad
        assert self.tokens[self.sos_index] == sos
        assert self.tokens[self.eos_index] == eos
        assert self.tokens[self.unknown_index] == unknown_token
   
    def __len__(self) -> int:
        return len(self.tokens)
   
    def token_to_index(self, token: str) -> int:
        return self.indices.get(token, self.unknown_index)
   
    def index_to_token(self, index: int) -> str:
        if 0 <= index < len(self):
            return self.tokens[index]
        return self.unknown_token

def build_vocabulary(
    words: Iterable[str],
    sos: str,
    eos: str,
    pad: str,
    unknown_token: str,
    max_size: int = None
) -> Vocab:
    """
    Build a vocabulary object from a text corpus with optional size limit.
    Special tokens (PAD, SOS, EOS, UNK) are always included regardless of max_size.
    
    Args:
        words: Iterable of words to build vocabulary from
        sos: Start of sequence token
        eos: End of sequence token
        pad: Padding token
        unknown_token: Token for unknown words
        max_size: Maximum vocabulary size (including special tokens)
    """
    # Use Counter instead of manual dictionary counting
    word_freq = Counter(words)
   
    # Sort by frequency - can use most_common() method
    if max_size is not None:
        # Reserve space for special tokens and get top words
        max_regular_tokens = max_size - 4  # PAD, SOS, EOS, UNK
        sorted_words = word_freq.most_common(max_regular_tokens)
    else:
        sorted_words = word_freq.most_common()
   
    # Create final token set
    tokens = set(word for word, _ in sorted_words)
    return Vocab(tokens, sos, eos, pad, unknown_token)