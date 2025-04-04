# Homework Assignment #2: The one with the RNNs

## Overview

This homework assignment focuses on developing skills to work with recurrent neural networks (RNNs) through two comprehensive tasks:

1. Implementing a basic Vanilla RNN and understanding key concepts like teacher forcing and warm start
2. Developing a sequence-to-sequence model for language translation

## Task 1: Vanilla RNN Implementation [5 points]

Implement a basic RNN cell and explore how teacher forcing and warm start affect prediction capabilities using a simple sine wave dataset.

### Implementation Requirements:
- Complete TODOs in the provided Jupyter notebook:
  - **TODO 1.1**: Implement one step of the forward pass for the RNN
  - **TODO 1.2**: Implement the complete forward step across an entire input sequence, including:
    - Teacher forcing during training
    - Warm start during evaluation

### Experiments:
Run the following configurations and analyze results:
1. Vary teacher forcing probability (p=0.0, p=0.5, p=0.75, p=1.0) with default values for other parameters
2. Keep teacher forcing at p=0.5 and test with very small and very large UNROLL_LENGTH values
3. Use UNROLL_LENGTH = 62 (maximum), teacher forcing off (p=0.0), and warm start of only 2
4. Use UNROLL_LENGTH = 3, teacher forcing on (p=1.0), and warm start of 2

### Analysis Questions:
1. Compare teacher forcing vs. learning on own samples:
   - What are the pros and cons of teacher forcing?
2. Identify which setup (combination of unroll_length and teacher forcing probability) causes the model to struggle
3. Explain how warm starting affects test time prediction and why
4. Analyze what happens when the structure of interest is much longer than the unroll length

### Key Parameters:
- **UNROLL_LENGTH**: Length of subsequences from sine wave used during training
- **TEACHER_FORCING_PROB**: Probability of applying teacher forcing
- **WARM_START**: Maximum number of timesteps considered in warm starting during prediction
- **NUM_ITERATIONS**, **LEARNING_RATE**, **REPORTING_INTERVAL**: Training control parameters

## Task 2: Sequence-to-Sequence Translation Model [5 points]

Develop a sequence-to-sequence architecture using LSTM or GRU layers for French-to-English translation.

### Dataset:
- **Multi30k**: ~30,000 parallel sentences in English, German, and French
- Average of ~12 words per sentence

### Implementation Requirements:
- Use spaCy language models for tokenization:
  - French: "fr_core_news_sm"
  - English: "en_core_web_sm"
- Leverage provided support code for:
  - Dataset loading and train/test splitting
  - Vocabulary creation with special tokens (sos, eos, unk, pad)
  - Sentence collation and padding

### Objective 1:
Implement and test a single layer LSTM encoder-decoder model for sequence-to-sequence translation, experimenting with:
- Input dropout after text embedding (p=0.1 or p=0.5)
- Word embedding dimensions (128, 256, 512)
- Hidden dimensions (128, 256, 512)
- Batch sizes (128, 256)
- Teacher forcing values (p=0.0, p=0.5, p=1.0)

### Objective 2:
For the best configuration from Objective 1, modify the encoder to use a bidirectional LSTM while keeping the decoder unidirectional.

### Required Results:
For each experiment in both objectives, provide:
- Training and test loss curves
- BLEU score and Perplexity metrics
- Discussion of key findings and justification of performance differences

## Architecture Overview

The basic sequence-to-sequence model consists of:
- Text embedding layers (one for encoder/French, one for decoder/English)
- Encoder implemented as LSTM (single layer initially, bidirectional in Objective 2)
- Decoder implemented as single layer LSTM
- Word Output Layer (Linear + Softmax) to convert decoder hidden state to vocabulary-sized vector

## Resources
- spaCy documentation for tokenization
- torchmetrics package for BLEU score and Perplexity calculation
- PyTorch documentation for LSTM and GRU implementation
