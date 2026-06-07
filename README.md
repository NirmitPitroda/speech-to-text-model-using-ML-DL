# Speech-to-Text Models using Deep Learning

> A progressive deep learning journey from raw data analysis to a full end-to-end Automatic Speech Recognition (ASR) system — built from scratch using PyTorch.

---

## Overview

This repository documents a complete, hands-on progression through machine learning and deep learning, culminating in an ASR system that transcribes speech to text. The project spans five stages: exploratory data analysis, classical ML, image classification with CNNs, audio classification with RNNs/LSTMs, and a production-grade ASR pipeline trained on LibriSpeech.

**The final model** uses a 3-layer Bidirectional LSTM trained with CTC loss on LibriSpeech `train-clean-100`, with beam search decoding enhanced by a KenLM n-gram language model for re-ranking — evaluated using Word Error Rate (WER) and Character Error Rate (CER).

---

## Repository Structure

```
speech-to-text-deep-learning/
│
├── 01_data_analysis/
│   └── vgchartz_eda.ipynb             # EDA on VGChartz 2024 sales dataset
│
├── 02_classical_ml/
│   ├── pokemon_type_classification.ipynb   # Multi-class classification on Pokémon stats
│   ├── pokemon_cas_regression.ipynb        # Linear Regression for attack score prediction
│   └── legendary_detection.ipynb          # Binary classifier + CAS histogram analysis
│
├── 03_image_classification/
│   ├── basic_neural_network.ipynb     # Feedforward NN on Flowers dataset (PyTorch)
│   └── cnn_classifier.ipynb           # CNN with data augmentation on Flowers dataset
│
├── 04_speech_command_recognition/
│   ├── rnn_classifier.ipynb           # RNN on Google Speech Commands (10 classes)
│   ├── bilstm_classifier.ipynb        # Bidirectional LSTM with dropout
│   └── cnn_lstm_hybrid.ipynb          # CNN + LSTM hybrid (bonus task)
│
└── 05_asr_model/
    └── lstm_asr.ipynb                 # Full ASR: BiLSTM + CTC + Beam Search + KenLM
```

---

## Stage-by-Stage Breakdown

### Stage 1 — Exploratory Data Analysis
**Notebook:** `01_data_analysis/vgchartz_eda.ipynb`

Foundational data analysis on the VGChartz 2024 video game sales dataset.

- Loaded, cleaned, and explored a real-world CSV dataset using Pandas and NumPy
- Engineered a `total_sales` feature by aggregating regional sales columns
- Computed statistical summaries (mean, median, std) using vectorized NumPy operations
- Visualized sales distributions and critic score vs sales relationships using Matplotlib and Seaborn

---

### Stage 2 — Classical Machine Learning
**Notebooks:** `02_classical_ml/`

Three distinct ML tasks on the Pokémon dataset.

**Task 1 — Type Classification (Multi-class)**
- Feature normalization and one-hot encoding of Pokémon types
- Trained and compared: Logistic Regression, KNN, Random Forest, SVM (RBF kernel), XGBoost
- Evaluated with accuracy, F1-score, and confusion matrix

**Task 2 — Regression (CAS Prediction)**
- Engineered a custom target: Combined Attack Score (`CAS = Attack + Sp.Atk + Speed × 0.5`)
- Trained a Linear Regression model excluding the constituent features
- Evaluated with MSE and predicted vs actual scatter plots

**Task 3 — Legendary Detection (Binary Classification)**
- Handled severe class imbalance (most Pokémon are non-legendary)
- Compared CAS histograms for legendary vs non-legendary Pokémon to identify hidden legendaries
- Built a binary classifier evaluated with accuracy, F1-score, and confusion matrix

---

### Stage 3 — Image Classification with Deep Learning
**Notebooks:** `03_image_classification/`

Progressive CNN development on the Flowers Recognition dataset (~4,317 images, 5 classes).

**Task 1 — Feedforward Neural Network**
- Images resized to 128×128 and flattened
- 2-layer dense network with ReLU activations and Softmax output
- Trained with CrossEntropyLoss and Adam optimizer

**Task 2 — Convolutional Neural Network**
- Custom `FlowerDataset` PyTorch class with `DataLoader`
- Architecture: `Conv2d(3→32) → ReLU → MaxPool → Conv2d(32→64) → ReLU → MaxPool → Flatten → Linear(65536→1024) → Linear(1024→128) → Linear(128→5)`
- Data augmentation: random horizontal flips, normalization
- Confusion matrix analysis across all 5 flower classes

---

### Stage 4 — Audio Classification (Speech Commands)
**Notebooks:** `04_speech_command_recognition/`

Trained three progressively more powerful models on 10 classes from the Google Speech Commands dataset (~38,500 samples @ 16kHz).

**Preprocessing (shared across all models)**
- 40-coefficient MFCC extraction using `torchaudio.transforms.MFCC` (`n_fft=400`, `hop_length=160`, `n_mels=40`)
- Per-sample feature standardization (zero mean, unit variance)
- Padding/truncation to fixed 16,000-sample length
- Data augmentation on training set: Gaussian noise injection, `FrequencyMasking`, `TimeMasking`
- Official train/validation/test splits from `validation_list.txt` and `testing_list.txt`

**Model 1 — Vanilla RNN** (`rnn_classifier.ipynb`)
- Single `nn.RNN` layer, hidden dim 128, last time-step pooling
- 10 epochs, Adam (lr=1e-3), CrossEntropyLoss
- Baseline for comparison

**Model 2 — Bidirectional LSTM** (`bilstm_classifier.ipynb`)
- `nn.LSTM` with `num_layers=2`, `hidden_dim=128`, `bidirectional=True`
- Concatenated final forward + backward hidden states → Dropout(0.3) → Linear
- 15 epochs — improved convergence and generalization over vanilla RNN

**Model 3 — CNN-LSTM Hybrid** (`cnn_lstm_hybrid.ipynb`) *(Bonus)*
- Input: Log-Mel Spectrograms treated as 2D images
- CNN block: `Conv2d(1→16, k=3) → ReLU → MaxPool(2,2) → Conv2d(16→32, k=3) → ReLU → MaxPool(2,2)`
- CNN output reshaped to `(B, T', C×M')` and fed into `nn.LSTM(hidden=128)`
- Final classification via `nn.Linear(128, 10)`
- 20 epochs — best performance of the three models

---

### Stage 5 — End-to-End ASR System
**Notebook:** `05_asr_model/lstm_asr.ipynb`  
**Platform:** Trained on Kaggle with NVIDIA Tesla T4 GPU

A full automatic speech recognition pipeline trained on LibriSpeech `train-clean-100` (~100 hours of clean English speech).

**Architecture**
```
Input: Log-Mel Spectrogram [B, T, 80]
         ↓
  3-layer Bidirectional LSTM (hidden_dim=256, bidirectional=True)
         ↓
  Output: [B, T, 512]
         ↓
  Linear Classifier → [B, T, vocab_size=28]
         ↓
  Log-Softmax (CTC-compatible)
```

**Vocabulary:** 28 characters — `<blank>`, space, and `a-z`

**Training**
- Loss: `nn.CTCLoss` (blank=space token, `zero_infinity=True`)
- Optimizer: Adam (lr=1e-4)
- Trained on a 5,000-sample subset for compute efficiency
- Batch size: 16, with custom `collate_fn` for variable-length sequence padding

| Epoch | CTC Loss |
|-------|----------|
| 1     | 0.8549   |
| 10    | 0.6295   |
| 20    | 0.4481   |
| 30    | 0.3115   |

Consistent convergence across 30 epochs, with loss reducing by ~64% from epoch 1 to epoch 30.

**Decoding**
- **Beam Search** (beam width=8): CTC-aware beam search with blank and repeated token handling
- **KenLM Language Model Re-ranking**: 3-gram pruned LM loaded via `kenlm`, combined with acoustic score as `0.1 × acoustic + 1.5 × LM_score`

**Evaluation** (on first batch after 30 epochs of training on 5k samples)

| Metric | Beam Search |
|--------|-------------|
| WER    | ~79%        |
| CER    | ~38%        |

> **Note on results:** The WER/CER reflect training on only 5,000 samples (a small subset of the full LibriSpeech train-clean-100 dataset of ~28,000 utterances), constrained by compute budget. The model shows clear learning — CER of ~38% on clean speech with a character-level BiLSTM trained from scratch in under 2 hours on a T4 GPU is a strong baseline. Training on the full dataset would substantially reduce both metrics.

---

## Model Architecture Summary

| Stage | Model | Dataset | Key Details |
|-------|-------|---------|-------------|
| 4 | Vanilla RNN | Speech Commands (10 cls) | 1-layer RNN, hidden=128, MFCC-40 |
| 4 | Bidirectional LSTM | Speech Commands (10 cls) | 2-layer BiLSTM, Dropout(0.3) |
| 4 | CNN-LSTM Hybrid | Speech Commands (10 cls) | 2× Conv2d + LSTM, Log-Mel input |
| 5 | ASR BiLSTM + CTC | LibriSpeech train-clean-100 | 3-layer BiLSTM, beam search + KenLM |

---

## Tech Stack

| Category | Tools |
|----------|-------|
| Deep Learning | PyTorch, torchaudio |
| Audio Processing | MFCC, Log-Mel Spectrograms, CTC Loss |
| Decoding | Beam Search (width=8), KenLM (`kenlm`) |
| Classical ML | scikit-learn (LogReg, KNN, RF, SVM), XGBoost |
| Evaluation | jiwer (WER/CER), sklearn metrics, confusion matrices |
| Data & Viz | NumPy, Pandas, Matplotlib, Seaborn |
| Platform | Kaggle (NVIDIA Tesla T4 GPU) |
| Datasets | VGChartz 2024, Pokémon (Kaggle), Flowers Recognition (Kaggle), Google Speech Commands, LibriSpeech |

---

## Setup

```bash
git clone https://github.com/nirmitpitroda/speech-to-text-deep-learning.git
cd speech-to-text-deep-learning
pip install -r requirements.txt
```

Open any notebook in Jupyter:
```bash
jupyter notebook
```

> **Note:** Stage 5 (`lstm_asr.ipynb`) was trained on Kaggle with GPU. The LibriSpeech dataset (~6GB) downloads automatically via `torchaudio.datasets.LIBRISPEECH`. The KenLM language model (`3-gram.pruned.3e-7.arpa`) is available from the OpenSLR project.

---

## Author

**Nirmit Pitroda**  
B.Tech, Electronics and Communication Engineering  
GitHub: [@nirmitpitroda](https://github.com/nirmitpitroda)
