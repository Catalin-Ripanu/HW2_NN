#!/usr/bin/env python3

"""
This module builds tokenizes the raw datasets and builds vocabularies.
"""

import os
import pathlib
import pickle
import multiprocessing as mp
from functools import partial
from typing import List, Tuple
import pandas as pd
import spacy
import spacy.language
import urllib.request
import tarfile
import shutil
import gzip
from itertools import islice

import constants, tokenization


def pad_sequences(
    sequences: list[list[str]], vocab: tokenization.Vocab, max_length: int
) -> list[list[str]]:
    """Pad sequences to max_length with PAD token and add SOS/EOS tokens."""
    padded_sequences = []
    for seq in sequences:
        # Add SOS and EOS tokens
        padded_seq = [constants.SOS] + seq + [constants.EOS]
        # Pad to max_length if necessary
        if len(padded_seq) < max_length:
            padded_seq.extend([constants.PAD] * (max_length - len(padded_seq)))
        padded_sequences.append(padded_seq)
    return padded_sequences


def process_batch(batch_data: Tuple[List[str], str, int]) -> List[List[str]]:
    """Process a batch of texts using spaCy."""
    texts, lang, max_length = batch_data
    nlp = spacy.load(f"{lang}_core_{'web' if lang == 'en' else 'news'}_sm", disable=['ner', 'parser'])
    # Use spaCy's built-in batch processing with optimized settings
    return [[tok.text for tok in doc] for doc in nlp.pipe(texts, batch_size=2000)]


def parallel_tokenize(
    texts: List[str],
    lang: str,
    max_length: int,
    batch_size: int = 2000,  # Increased batch size
    n_processes: int = None,
) -> List[List[str]]:
    """Tokenize texts in parallel using multiple processes."""
    if n_processes is None:
        n_processes = min(mp.cpu_count(), 4)  # Limit max processes

    # Split texts into larger batches for better efficiency
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    batch_data = [(batch, lang, max_length) for batch in batches]

    # Use context manager with maxtasksperchild to prevent memory leaks
    with mp.Pool(n_processes, maxtasksperchild=1000) as pool:
        results = pool.map(process_batch, batch_data, chunksize=1)

    return [tokens for batch in results for tokens in batch]


def parse_training(
    train_path: pathlib.Path, lang: str, max_length: int = 128
) -> Tuple[List[List[str]], tokenization.Vocab]:
    """Parse training data using parallel processing."""
    # Read file in chunks to reduce memory usage
    chunk_size = 10000
    texts = []
    with open(train_path, "r", encoding="utf-8") as fin:
        while True:
            chunk = list(islice(fin, chunk_size))
            if not chunk:
                break
            texts.extend([line.strip() for line in chunk])

    parsed = parallel_tokenize(texts, lang, max_length)
    
    # Build vocabulary using generator to save memory
    vocab = tokenization.build_vocabulary(
        words=(word for text in parsed for word in text),
        sos=constants.SOS,
        eos=constants.EOS,
        pad=constants.PAD,
        unknown_token=constants.UNKNOWN,
    )

    return parsed, vocab


def parse_validation(
    valid_path: pathlib.Path, lang: str, max_length: int = 128
) -> List[List[str]]:
    """Parse validation data using parallel processing."""
    with open(valid_path, "r", encoding="utf-8") as fin:
        texts = fin.read().splitlines()
    return parallel_tokenize(texts, lang, max_length)


def save_parquet(df: pd.DataFrame, path: pathlib.Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    df.to_parquet(path)


def save_vocab(vocab: tokenization.Vocab, path: pathlib.Path) -> None:
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "wb") as fout:
        pickle.dump(vocab, fout)


def download_and_decompress(url: str, output_path: str) -> bool:
    """
    Downloads a gzipped file and decompresses it to the specified output path.

    Args:
        url: URL of the gzipped file to download
        output_path: Path where the decompressed file should be saved

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create a temporary file for the compressed data
        gz_path = output_path + ".gz"

        # Download the compressed file
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, gz_path)

        # Decompress the file
        print(f"Decompressing {gz_path}...")
        with gzip.open(gz_path, "rb") as f_in:
            with open(output_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Clean up the compressed file
        os.remove(gz_path)
        print(f"Successfully downloaded and decompressed to {output_path}")
        return True

    except Exception as e:
        print(f"Error processing {url}: {e}")
        # Clean up any partial downloads
        if os.path.exists(gz_path):
            os.remove(gz_path)
        if os.path.exists(output_path):
            os.remove(output_path)
        return False


def download_multi30k() -> bool:
    """
    Downloads and extracts the Multi30K dataset files.
    Creates necessary directory structure and places files in the correct locations.
    """
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/tokenized", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # URLs for the dataset files
    urls = {
        "train.en": "https://github.com/multi30k/dataset/raw/master/data/task1/raw/train.en.gz",
        "train.fr": "https://github.com/multi30k/dataset/raw/master/data/task1/raw/train.fr.gz",
        "val.en": "https://github.com/multi30k/dataset/raw/master/data/task1/raw/val.en.gz",
        "val.fr": "https://github.com/multi30k/dataset/raw/master/data/task1/raw/val.fr.gz",
    }

    # Download and decompress each file
    success = True
    for filename, url in urls.items():
        output_path = f"data/raw/{filename}"
        if not os.path.exists(output_path):
            if not download_and_decompress(url, output_path):
                success = False
                break

    return success


def prepare_datasets() -> bool:
    """
    Prepares the Multi30K dataset and sets up the environment.
    Returns True if successful, False otherwise.
    """
    # First download and decompress the dataset
    if not download_multi30k():
        print("Failed to download and decompress dataset files")
        return False

    # Verify that all required files exist and are not empty
    required_files = [
        "data/raw/train.en",
        "data/raw/train.fr",
        "data/raw/val.en",
        "data/raw/val.fr",
    ]

    for file in required_files:
        if not os.path.exists(file):
            print(f"Missing required file: {file}")
            return False
        if os.path.getsize(file) == 0:
            print(f"File is empty: {file}")
            return False

    print("Dataset setup completed successfully!")
    return True


def pad_batch(batch: List[List[str]], max_length: int) -> List[List[str]]:
    """Pad a batch of sequences."""
    return [
        [constants.SOS]
        + seq
        + [constants.EOS]
        + [constants.PAD] * (max_length - len(seq) - 2)
        for seq in batch
    ]


def parallel_pad_sequences(
    sequences: List[List[str]],
    vocab: tokenization.Vocab,
    max_length: int,
    batch_size: int = 1000,
) -> List[List[str]]:
    """Pad sequences in parallel."""
    # Split into batches
    batches = [
        sequences[i : i + batch_size] for i in range(0, len(sequences), batch_size)
    ]

    # Process in parallel
    with mp.Pool() as pool:
        # Create a partial function with max_length
        pad_func = partial(pad_batch, max_length=max_length)
        padded_batches = pool.map(pad_func, batches)

    # Flatten results
    return [seq for batch in padded_batches for seq in batch]


def get_vocabularies(max_length: int = 128):
    """Process datasets with parallel processing."""
    # Ensure dataset exists
    if not os.path.exists("data/raw/train.en"):
        print("Dataset not found. Downloading Multi30K dataset...")
        if not prepare_datasets():
            raise RuntimeError(
                "Failed to prepare dataset. Please check error messages above."
            )

    # Process English and French data in parallel
    print("Processing English training data...")
    train_en, vocab_en = parse_training(
        pathlib.Path("data/raw/train.en"), "en", max_length
    )
    print("Processing English validation data...")
    valid_en = parse_validation(pathlib.Path("data/raw/val.en"), "en", max_length)

    print("Processing French training data...")
    train_fr, vocab_fr = parse_training(
        pathlib.Path("data/raw/train.fr"), "fr", max_length
    )
    print("Processing French validation data...")
    valid_fr = parse_validation(pathlib.Path("data/raw/val.fr"), "fr", max_length)

    # Parallel padding
    print("Padding sequences...")
    train_en = parallel_pad_sequences(train_en, vocab_en, max_length)
    train_fr = parallel_pad_sequences(train_fr, vocab_fr, max_length)
    valid_en = parallel_pad_sequences(valid_en, vocab_en, max_length)
    valid_fr = parallel_pad_sequences(valid_fr, vocab_fr, max_length)

    # Save results
    print("Saving processed data...")
    train_path = pathlib.Path("data/tokenized/train.parquet")
    df_train = pd.DataFrame({"en": train_en, "fr": train_fr})
    save_parquet(df_train, train_path)

    valid_path = pathlib.Path("data/tokenized/valid.parquet")
    df_valid = pd.DataFrame({"en": valid_en, "fr": valid_fr})
    save_parquet(df_valid, valid_path)

    en_path = pathlib.Path("models/vocab_en.pkl")
    save_vocab(vocab_en, en_path)

    fr_path = pathlib.Path("models/vocab_fr.pkl")
    save_vocab(vocab_fr, fr_path)

    print("Processing completed successfully!")
    return vocab_en, vocab_fr, df_train, df_valid
