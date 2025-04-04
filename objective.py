import os
from torchmetrics.text import BLEUScore
import math
import seaborn as sns
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pathlib
import pickle
from torch.utils.data import DataLoader
import random
import numpy as np
from typing import Dict, Tuple, List, Union
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from itertools import product
import json
from datetime import datetime
from tqdm import tqdm

import constants
from dataload import Multi30kDataset, make_collator
from prepare_datasets import get_vocabularies
import tokenization

EPOCHS = 25

class MetricsTracker:
    def __init__(self, vocab_en, vocab_fr):
        self.vocab_en = vocab_en
        self.vocab_fr = vocab_fr
        self.bleu = BLEUScore(
            n_gram=2,
            smooth=True,
        )
        self.metrics_history = defaultdict(list)

    def compute_perplexity(self, loss: float) -> float:
        """Compute perplexity from cross-entropy loss."""
        return math.exp(loss)

    def decode_sequence(self, seq: Union[torch.Tensor, List[int]]) -> List[str]:
        """
        Convert token indices to words.

        Args:
            seq: Either a torch.Tensor or a List[int] containing token indices

        Returns:
            List[str]: List of decoded words
        """
        words = []
        for idx in seq:
            # Handle both tensor and integer inputs
            token_idx = idx.item() if isinstance(idx, torch.Tensor) else idx
            token = self.vocab_en.index_to_token(token_idx)
            if token in [constants.PAD, constants.SOS, constants.EOS]:
                continue
            words.append(token)
        return words

    def compute_bleu(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute BLEU score for batch of predictions."""
        pred_texts = [self.decode_sequence(pred) for pred in predictions]
        target_texts = [self.decode_sequence(target) for target in targets]

        # Filter out empty predictions/targets and very short sequences
        valid_pairs = [(p, t) for p, t in zip(pred_texts, target_texts) 
                      if len(p) > 1 and len(t) > 1]
        
        if not valid_pairs:
            return 0.0

        pred_texts, target_texts = zip(*valid_pairs)

        # Convert to list of strings for BLEU calculation
        pred_texts = [" ".join(text) for text in pred_texts]
        target_texts = [[" ".join(text)] for text in target_texts]

        try:
            return self.bleu(pred_texts, target_texts).item()
        except Exception as e:
            print(f"Error computing BLEU: {e}")
            print(f"Sample pred: {pred_texts[0] if pred_texts else 'None'}")
            print(f"Sample target: {target_texts[0] if target_texts else 'None'}")
            return 0.0

    def update(self, metric_name: str, value: float):
        """Update metrics history."""
        self.metrics_history[metric_name].append(value)


class ExperimentAnalyzer:
    def __init__(self, experiment_dir: pathlib.Path):
        self.experiment_dir = experiment_dir
        self.results = []

    def add_experiment_result(self, config: Dict, metrics: Dict):
        """Add results from a single experiment."""
        result = {"config": config, "metrics": metrics}
        self.results.append(result)

    def plot_training_curves(self, exp_idx: int, metrics: MetricsTracker):
        """Plot training and validation curves with metrics."""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))

        # Loss curves
        ax1.plot(metrics.metrics_history["train_loss"], label="Training Loss")
        ax1.plot(metrics.metrics_history["valid_loss"], label="Validation Loss")
        ax1.set_title("Training and Validation Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()

        # Perplexity curves
        ax2.plot(
            metrics.metrics_history["train_perplexity"], label="Training Perplexity"
        )
        ax2.plot(
            metrics.metrics_history["valid_perplexity"], label="Validation Perplexity"
        )
        ax2.set_title("Training and Validation Perplexity")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Perplexity")
        ax2.legend()

        # BLEU score curve
        ax3.plot(metrics.metrics_history["bleu_score"], label="BLEU Score")
        ax3.set_title("BLEU Score Progress")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("BLEU Score")
        ax3.legend()

        plt.tight_layout()
        plt.savefig(self.experiment_dir / f"experiment_{exp_idx}_metrics.png")
        plt.close()

    def create_summary_plots(self):
        """Create summary plots comparing different configurations."""
        df = pd.DataFrame(self.results)
        df = pd.concat(
            [pd.json_normalize(df["config"]), pd.json_normalize(df["metrics"])], axis=1
        )

        # Create plots for each hyperparameter
        params = [
            "emb_dim",
            "hid_dim",
            "batch_size",
            "teacher_forcing_ratio",
            "emb_dropout",
        ]
        metrics = ["final_bleu", "best_valid_perplexity", "best_valid_loss"]

        for metric in metrics:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            fig.suptitle(f"Impact of Hyperparameters on {metric}")

            for i, param in enumerate(params):
                ax = axes[i // 3, i % 3]
                sns.boxplot(data=df, x=param, y=metric, ax=ax)
                ax.set_title(f"{param} vs {metric}")
                ax.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig(self.experiment_dir / f"summary_{metric}_analysis.png")
            plt.close()

        # Create correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(df[params + metrics].corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation between Parameters and Metrics")
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "correlation_heatmap.png")
        plt.close()

        # Save detailed results
        df.to_csv(self.experiment_dir / "detailed_results.csv", index=False)

        return df

    def generate_report(self, df: pd.DataFrame):
        """Generate a detailed analysis report."""
        report = []

        # Best configurations for each metric
        metrics = ["final_bleu", "best_valid_perplexity", "best_valid_loss"]
        report.append("Best Configurations:")
        for metric in metrics:
            best_idx = (
                df[metric].argmax() if metric == "final_bleu" else df[metric].argmin()
            )
            best_config = df.iloc[best_idx]

            report.append(f"\nBest {metric}:")
            report.append(f"Value: {best_config[metric]:.4f}")
            report.append("Configuration:")
            for param in [
                "emb_dim",
                "hid_dim",
                "batch_size",
                "teacher_forcing_ratio",
                "emb_dropout",
            ]:
                report.append(f"- {param}: {best_config[param]}")

        # Parameter impact analysis
        report.append("\nParameter Impact Analysis:")
        for param in [
            "emb_dim",
            "hid_dim",
            "batch_size",
            "teacher_forcing_ratio",
            "emb_dropout",
        ]:
            report.append(f"\n{param} analysis:")
            for metric in metrics:
                grouped = df.groupby(param)[metric].agg(["mean", "std"])
                best_value = (
                    grouped["mean"].idxmin()
                    if metric != "final_bleu"
                    else grouped["mean"].idxmax()
                )
                report.append(f"- Impact on {metric}:")
                report.append(f"  Best value: {best_value}")
                report.append(f"  Mean performance: {grouped['mean'][best_value]:.4f}")
                report.append(f"  Std deviation: {grouped['std'][best_value]:.4f}")

        # Save report
        with open(self.experiment_dir / "analysis_report.txt", "w") as f:
            f.write("\n".join(report))

        return "\n".join(report)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        emb_dim: int,
        hid_dim: int,
        n_layers: int,
        emb_dropout: float,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.n_layers = n_layers
        self.hidden_dim = hid_dim
        self.bidirectional = bidirectional

        # Bidirectional LSTM for Objective 2 if bidirectional = True
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        # Linear layer to reduce bidirectional output to expected decoder size
        self.fc = nn.Linear(hid_dim * 2 if bidirectional else hid_dim, hid_dim)

    def forward(
        self, src: torch.Tensor, src_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        embedded = self.emb_dropout(self.embedding(src))

        src_lengths, sort_idx = src_lengths.sort(descending=True)
        embedded = embedded[sort_idx]

        try:
            packed_embedded = pack_padded_sequence(
                embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=True
            )

            packed_outputs, (hidden, cell) = self.lstm(packed_embedded)

            outputs, _ = pad_packed_sequence(
                packed_outputs, batch_first=True, padding_value=0.0
            )

            _, unsort_idx = sort_idx.sort()
            outputs = outputs[unsort_idx]

            # Handle bidirectional hidden states
            if self.bidirectional:
                # Reshape hidden and cell states
                hidden = hidden.view(self.n_layers, 2, -1, self.hidden_dim)
                cell = cell.view(self.n_layers, 2, -1, self.hidden_dim)

                # Combine bidirectional states
                hidden = torch.cat([hidden[:, 0], hidden[:, 1]], dim=2)
                cell = torch.cat([cell[:, 0], cell[:, 1]], dim=2)

                # Project concatenated states to decoder size
                hidden = hidden.view(-1, hidden.size(2))
                cell = cell.view(-1, cell.size(2))
                
                hidden = self.fc(hidden)
                cell = self.fc(cell)
                
                # Reshape back to (n_layers, batch, hidden_dim)
                hidden = hidden.view(self.n_layers, -1, self.hidden_dim)
                cell = cell.view(self.n_layers, -1, self.hidden_dim)

            hidden = hidden[:, unsort_idx]
            cell = cell[:, unsort_idx]

            return outputs, (hidden, cell)

        except RuntimeError as e:
            print(f"Error during packing/unpacking:")
            print(f"src_lengths min: {src_lengths.min().item()}")
            print(f"src_lengths max: {src_lengths.max().item()}")
            print(f"embedded shape: {embedded.shape}")
            raise e


class Decoder(nn.Module):
    def __init__(
        self,
        output_dim: int,
        emb_dim: int,
        hid_dim: int,
        n_layers: int,
        emb_dropout: float,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hid_dim

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Add num_layers parameter and use hidden_dropout in LSTM
        self.lstm = nn.LSTM(
            emb_dim,
            hid_dim,
            num_layers=n_layers,
            batch_first=True,
        )
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(
        self, input: torch.Tensor, hidden: torch.Tensor, cell: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input = input.unsqueeze(1)  # Add sequence length dimension
        embedded = self.emb_dropout(self.embedding(input))

        # Ensure hidden state dimensions match
        assert (
            hidden.size(0) == self.n_layers
        ), f"Expected hidden state with {self.n_layers} layers, got {hidden.size(0)}"

        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(
        self,
        src: torch.Tensor,
        src_lengths: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        _, (hidden, cell) = self.encoder(src, src_lengths)
        input = trg[:, 0]

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1

        return outputs


class ExperimentConfig:
    def __init__(self):
        # Base configuration
        self.base_emb_dim = 128
        self.base_hid_dim = 128
        self.base_batch_size = 128
        self.base_teacher_forcing = 0.0
        self.base_emb_dropout = 0.5
        self.base_num_layers = 1

        # Parameter variations
        self.emb_dims = [128, 256, 512]
        self.hid_dims = [128, 256, 512]
        self.batch_sizes = [128, 256]
        self.teacher_forcing_ratios = [0.0, 0.5, 1.0]
        self.emb_dropouts = [0.0, 0.5, 1.0]

    def generate_configs(self) -> List[Dict]:
        configs = []
        
        # Helper function to create base config
        def create_base_config():
            return {
                "emb_dim": self.base_emb_dim,
                "hid_dim": self.base_hid_dim,
                "batch_size": self.base_batch_size,
                "teacher_forcing_ratio": self.base_teacher_forcing,
                "emb_dropout": self.base_emb_dropout,
                "num_layers": self.base_num_layers,
                "n_epochs": EPOCHS,
                "clip": 1.0,
                "learning_rate": 0.001,
                "bidirectional": False,
            }

        # Vary batch size
        for batch_size in self.batch_sizes:
            config = create_base_config()
            config["batch_size"] = batch_size
            configs.append(config.copy())

        # Vary teacher forcing ratio
        for tf_ratio in self.teacher_forcing_ratios:
            config = create_base_config()
            config["teacher_forcing_ratio"] = tf_ratio
            configs.append(config.copy())

        # Vary embedding dropout
        for emb_dropout in self.emb_dropouts:
            config = create_base_config()
            config["emb_dropout"] = emb_dropout
            configs.append(config.copy())

        return configs


class TranslationTrainer:
    def __init__(
        self, config: Dict, vocab_en: tokenization.Vocab, vocab_fr: tokenization.Vocab
    ):
        self.config = config
        self.vocab_en = vocab_en
        self.vocab_fr = vocab_fr

        # Set up multi-GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # Limit number of GPUs to 3 for optimal performance
        if self.n_gpus > 3:
            print(f"Limiting GPU usage to 3 GPUs for optimal performance (was {self.n_gpus})")
            self.n_gpus = 3
            # Set visible devices to only first 3 GPUs
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

        if self.n_gpus > 1:
            print(f"Using {self.n_gpus} GPUs!")
            # Adjust batch size for multi-GPU
            self.config["batch_size"] = self.config["batch_size"] * self.n_gpus
            
            # Ensure minimum batch size per GPU
            min_batch_per_gpu = 16
            if self.config["batch_size"] // self.n_gpus < min_batch_per_gpu:
                self.config["batch_size"] = min_batch_per_gpu * self.n_gpus
                print(f"Adjusted batch size to {self.config['batch_size']} to ensure minimum {min_batch_per_gpu} samples per GPU")

        # Create model
        encoder = Encoder(
            len(vocab_fr),
            config["emb_dim"],
            config["hid_dim"],
            config["num_layers"],
            config["emb_dropout"],
            config["bidirectional"],
        )
        decoder = Decoder(
            len(vocab_en),
            config["emb_dim"],
            config["hid_dim"],
            config["num_layers"],
            config["emb_dropout"],
        )
        self.model = Seq2Seq(encoder, decoder, self.device)

        # Move model to device first
        self.model = self.model.to(self.device)
        
        # Then wrap with DataParallel if multiple GPUs
        if self.n_gpus > 1:
            self.model = nn.DataParallel(self.model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=config["learning_rate"])

        # Enable cuDNN benchmarking and deterministic mode
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Faster training, slightly less reproducible

        # Use mixed precision training
        self.scaler = torch.cuda.amp.GradScaler()

        # Move criterion to GPU
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=vocab_en.token_to_index(constants.PAD), 
            reduction="mean"
        ).to(self.device)

        self.train_losses = []
        self.valid_losses = []
        self.metrics = MetricsTracker(vocab_en, vocab_fr)

        # Add gradient accumulation steps
        self.gradient_accumulation_steps = 1
        
        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=2
        )

        # Enable torch.backends.cuda.matmul.allow_tf32 for faster matrix multiplications
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    def train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_batches = len(train_loader)
        running_loss = 0.0
        total_correct = 0
        total_tokens = 0
        pad_idx = self.vocab_en.token_to_index(constants.PAD)

        # Use torch.cuda.Stream for overlapping compute and data transfer
        data_stream = torch.cuda.Stream()
        
        train_pbar = tqdm(
            train_loader,
            total=total_batches,
            desc=f"Epoch {epoch+1}/{self.config['n_epochs']} [Train]",
            leave=True,
        )

        # Pre-allocate tensors for accumulating gradients
        if self.n_gpus > 1:
            for param in self.model.module.parameters():
                if param.requires_grad:
                    param.grad = torch.zeros_like(param)
        else:
            for param in self.model.parameters():
                if param.requires_grad:
                    param.grad = torch.zeros_like(param)

        for batch_idx, (src, src_lengths, trg) in enumerate(train_pbar):
            torch.cuda.synchronize()
            with torch.cuda.stream(data_stream):
                # Async transfer to GPU
                src = src.to(self.device, non_blocking=True)
                src_lengths = src_lengths.to(self.device, non_blocking=True)
                trg = trg.to(self.device, non_blocking=True)

            # Only zero gradients at the start of accumulation steps
            if (batch_idx % self.gradient_accumulation_steps) == 0:
                self.optimizer.zero_grad(set_to_none=True)

            try:
                with torch.cuda.amp.autocast():
                    if self.n_gpus > 1:
                        output = self.model.module(
                            src, src_lengths, trg, self.config["teacher_forcing_ratio"]
                        )
                    else:
                        output = self.model(
                            src, src_lengths, trg, self.config["teacher_forcing_ratio"]
                        )

                    target = trg[:, 1:].contiguous()
                    output = output[:, :-1, :].contiguous()

                    batch_size, seq_len, vocab_size = output.shape
                    output_flat = output.view(-1, vocab_size)
                    target_flat = target.view(-1)

                    non_pad_mask = target_flat.ne(pad_idx)
                    
                    loss = self.criterion(
                        output_flat[non_pad_mask],
                        target_flat[non_pad_mask]
                    )
                    # Scale loss for gradient accumulation
                    loss = loss / self.gradient_accumulation_steps

                # Scale loss and backward pass
                self.scaler.scale(loss).backward()
                
                # Only update weights after accumulating gradients
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    # Unscale gradients and clip
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config["clip"]
                    )
                    
                    # Update weights
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                # Calculate accuracy (outside of autocast)
                with torch.no_grad():
                    predicted = output.argmax(dim=-1)
                    correct = predicted.eq(target).masked_select(target.ne(pad_idx)).sum().item()
                    tokens = non_pad_mask.sum().item()

                    total_correct += correct
                    total_tokens += tokens

                    accuracy = 100 * total_correct / total_tokens if total_tokens > 0 else 0
                    running_loss += loss.item()

                train_pbar.set_postfix(
                    {
                        "loss": f"{running_loss/(batch_idx+1):.3f}",
                        "acc": f"{accuracy:.2f}%",
                        "batch": f"{batch_idx+1}/{total_batches}",
                    }
                )

            except RuntimeError as e:
                print(f"Error in batch {batch_idx}:")
                print(f"src shape: {src.shape}")
                print(f"src_lengths shape: {src_lengths.shape}")
                print(f"trg shape: {trg.shape}")
                print(f"output shape: {output.shape}")
                raise e

        # Update learning rate based on validation loss
        self.scheduler.step(running_loss / total_batches)
        
        train_pbar.close()
        return running_loss / total_batches

    def evaluate(self, valid_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        self.model.eval()
        total_batches = len(valid_loader)
        running_loss = 0.0
        total_correct = 0
        total_tokens = 0
        pad_idx = self.vocab_en.token_to_index(constants.PAD)

        # Store predictions and targets for BLEU score
        all_predictions = []
        all_targets = []

        valid_pbar = tqdm(
            valid_loader,
            total=total_batches,
            desc=f"Epoch {epoch+1}/{self.config['n_epochs']} [Valid]",
            leave=True,
        )

        with torch.no_grad():
            for batch_idx, (src, src_lengths, trg) in enumerate(valid_pbar):
                # Move tensors to GPU with non_blocking=True
                src = src.to(self.device, non_blocking=True)
                src_lengths = src_lengths.to(self.device, non_blocking=True)
                trg = trg.to(self.device, non_blocking=True)

                try:
                    # Handle DataParallel wrapper
                    if self.n_gpus > 1:
                        output = self.model.module(src, src_lengths, trg, 0)  # No teacher forcing during evaluation
                    else:
                        output = self.model(src, src_lengths, trg, 0)  # No teacher forcing during evaluation

                    # Prepare target for loss computation (remove SOS token)
                    target = trg[:, 1:].contiguous()
                    # Remove last prediction (corresponding to EOS token)
                    output = output[:, :-1, :].contiguous()

                    # Reshape for loss computation
                    batch_size, seq_len, vocab_size = output.shape
                    output_flat = output.view(-1, vocab_size)
                    target_flat = target.view(-1)

                    # Create mask for non-padding tokens
                    non_pad_mask = target_flat.ne(pad_idx)
                    
                    # Compute loss only on non-padding tokens
                    loss = self.criterion(
                        output_flat[non_pad_mask],
                        target_flat[non_pad_mask]
                    )
                    running_loss += loss.item()

                    # Get predicted tokens and calculate accuracy
                    predicted = output.argmax(dim=-1)
                    correct = predicted.eq(target).masked_select(target.ne(pad_idx)).sum().item()
                    tokens = non_pad_mask.sum().item()

                    total_correct += correct
                    total_tokens += tokens

                    accuracy = 100 * total_correct / total_tokens if total_tokens > 0 else 0

                    # Store predictions and targets for BLEU score
                    # Remove padding tokens before storing
                    for pred, targ in zip(predicted, target):
                        # Get mask for non-padding tokens
                        mask = targ.ne(pad_idx)
                        # Get only the non-padding tokens
                        pred_tokens = pred[mask].cpu().tolist()
                        targ_tokens = targ[mask].cpu().tolist()
                        all_predictions.append(pred_tokens)
                        all_targets.append(targ_tokens)

                    valid_pbar.set_postfix(
                        {
                            "loss": f"{running_loss/(batch_idx+1):.3f}",
                            "acc": f"{accuracy:.2f}%",
                            "batch": f"{batch_idx+1}/{total_batches}",
                        }
                    )

                except RuntimeError as e:
                    print(f"Error in batch {batch_idx}:")
                    print(f"src shape: {src.shape}")
                    print(f"src_lengths shape: {src_lengths.shape}")
                    print(f"trg shape: {trg.shape}")
                    print(f"output shape: {output.shape}")
                    raise e

        valid_pbar.close()

        # Calculate average loss and BLEU score
        avg_loss = running_loss / total_batches
        bleu_score = self.metrics.compute_bleu(all_predictions, all_targets)

        return avg_loss, bleu_score

    def train(self, train_loader: DataLoader, valid_loader: DataLoader) -> Dict:
        best_valid_loss = float("inf")
        experiment_results = {
            "config": self.config,
            "metrics": {},
        }

        for epoch in range(self.config["n_epochs"]):
            train_loss = self.train_epoch(train_loader, epoch)
            train_perplexity = self.metrics.compute_perplexity(train_loss)
            valid_loss, bleu_score = self.evaluate(valid_loader, epoch)
            valid_perplexity = self.metrics.compute_perplexity(valid_loss)

            # Update metrics
            self.metrics.update("train_loss", train_loss)
            self.metrics.update("valid_loss", valid_loss)
            self.metrics.update("train_perplexity", train_perplexity)
            self.metrics.update("valid_perplexity", valid_perplexity)
            self.metrics.update("bleu_score", bleu_score)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                experiment_results["best_valid_loss"] = best_valid_loss

                # Save model with config details in filename
                model_path = (
                    f'models/model_emb{self.config["emb_dim"]}_'
                    f'hid{self.config["hid_dim"]}_'
                    f'tf{self.config["teacher_forcing_ratio"]}_'
                    f'drop{self.config["emb_dropout"]}.pt'
                )
                pathlib.Path('models').mkdir(exist_ok=True)  # Ensure directory exists
                
                if isinstance(self.model, nn.DataParallel):
                    torch.save(self.model.module.state_dict(), model_path)
                else:
                    torch.save(self.model.state_dict(), model_path)

            print(f"Epoch: {epoch+1}/{self.config['n_epochs']}")
            print(
                f"\tTrain Loss: {train_loss:.3f} (Perplexity: {train_perplexity:.3f})"
            )
            print(
                f"\tValid Loss: {valid_loss:.3f} (Perplexity: {valid_perplexity:.3f})"
            )
            print(f"\tBLEU Score: {bleu_score:.3f}")

        experiment_results["metrics"] = {
            "final_bleu": self.metrics.metrics_history["bleu_score"][-1],
            "best_valid_perplexity": min(
                self.metrics.metrics_history["valid_perplexity"]
            ),
            "best_valid_loss": min(self.metrics.metrics_history["valid_loss"]),
            "history": dict(self.metrics.metrics_history),
        }

        return experiment_results


def run_experiments():
    # Set random seed
    set_seed(42)

    # Generate experimental configurations
    experiment_configs = ExperimentConfig().generate_configs()

    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = pathlib.Path(f"experiments_{timestamp}")
    experiment_dir.mkdir(exist_ok=True)

    analyzer = ExperimentAnalyzer(experiment_dir)

    best_config = None
    best_bleu = 0.0  # Initialize to 0.0 explicitly

    print("Objective_1")
    for i, config in enumerate(experiment_configs):
        print(f"\nRunning experiment {i+1}/{len(experiment_configs)}")  # Add total count
        print("Configuration:", json.dumps(config, indent=2))  # Better formatting

        # Load data
        vocab_en, vocab_fr, train_df, valid_df = get_vocabularies(config["hid_dim"])

        # Create datasets and dataloaders
        train_dataset = Multi30kDataset(train_df, vocab_en, vocab_fr)
        valid_dataset = Multi30kDataset(valid_df, vocab_en, vocab_fr)

        collate_fn = make_collator(vocab_en, vocab_fr)

        # Calculate effective batch size based on GPU count
        n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        effective_batch_size = config["batch_size"] // n_gpus if n_gpus > 1 else config["batch_size"]

        train_loader = DataLoader(
            train_dataset,
            batch_size=effective_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=5 * n_gpus,  # Scale workers with GPU count
            pin_memory=True,
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=effective_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=5 * n_gpus,  # Scale workers with GPU count
            pin_memory=True,
        )

        # Train model
        trainer = TranslationTrainer(config, vocab_en, vocab_fr)
        results = trainer.train(train_loader, valid_loader)

        # Plot individual experiment metrics
        analyzer.plot_training_curves(i, trainer.metrics)

        # Add results to analyzer
        analyzer.add_experiment_result(config, results["metrics"])

        # Update best configuration
        current_bleu = results["metrics"]["final_bleu"]
        if current_bleu > best_bleu:
            best_bleu = current_bleu
            best_config = config.copy()  # Make a copy to avoid reference issues

    # Ensure best_config exists
    if best_config is None:
        raise RuntimeError("No experiments completed successfully")

    print("\nObjective_2")
    print("Best Configuration from Objective 1:", json.dumps(best_config, indent=2))

    # Enable bidirectional for best config
    best_config["bidirectional"] = True

    # Load data for best config
    vocab_en, vocab_fr, train_df, valid_df = get_vocabularies(best_config["hid_dim"])

    # Create datasets and dataloaders
    train_dataset = Multi30kDataset(train_df, vocab_en, vocab_fr)
    valid_dataset = Multi30kDataset(valid_df, vocab_en, vocab_fr)

    collate_fn = make_collator(vocab_en, vocab_fr)

    # Calculate effective batch size based on GPU count
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch_size = best_config["batch_size"] // n_gpus if n_gpus > 1 else best_config["batch_size"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4 * n_gpus,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4 * n_gpus,
        pin_memory=True,
    )

    # Train model with bidirectional encoder
    trainer = TranslationTrainer(best_config, vocab_en, vocab_fr)
    results = trainer.train(train_loader, valid_loader)

    # Plot final experiment metrics
    analyzer.plot_training_curves(len(experiment_configs), trainer.metrics)
    analyzer.add_experiment_result(best_config, results["metrics"])

    # Generate and save analysis
    results_df = analyzer.create_summary_plots()
    report = analyzer.generate_report(results_df)

    print("\nExperiment Analysis Report:")
    print(report)


if __name__ == "__main__":
    run_experiments()
