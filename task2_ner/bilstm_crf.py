"""
Task 2: BiLSTM-CRF Model for Named Entity Recognition
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchcrf import CRF
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE


class NERDataset(Dataset):
    """Dataset for NER sequences."""
    def __init__(self, token_ids, tag_ids, lengths):
        self.token_ids = torch.LongTensor(token_ids)
        self.tag_ids = torch.LongTensor(tag_ids)
        self.lengths = torch.LongTensor(lengths)

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, idx):
        return self.token_ids[idx], self.tag_ids[idx], self.lengths[idx]


class BiLSTMCRF(nn.Module):
    """BiLSTM-CRF model for sequence labeling."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_tags, dropout=0.5, padding_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                            bidirectional=True, num_layers=2, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.hidden2tag = nn.Linear(hidden_dim * 2, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def _get_emissions(self, x):
        embedded = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        emissions = self.hidden2tag(lstm_out)
        return emissions

    def forward(self, x, tags, mask):
        """Compute negative log-likelihood loss."""
        emissions = self._get_emissions(x)
        loss = -self.crf(emissions, tags, mask=mask, reduction="mean")
        return loss

    def decode(self, x, mask):
        """Viterbi decode to get best tag sequence."""
        emissions = self._get_emissions(x)
        return self.crf.decode(emissions, mask=mask)


def train_bilstm_crf(model, train_loader, val_loader, epochs, lr, device=DEVICE):
    """Train the BiLSTM-CRF model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for token_ids, tag_ids, lengths in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            token_ids = token_ids.to(device)
            tag_ids = tag_ids.to(device)
            lengths = lengths.to(device)

            # Create mask: True where tokens exist (not padding)
            max_len = token_ids.size(1)
            mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

            optimizer.zero_grad()
            loss = model(token_ids, tag_ids, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for token_ids, tag_ids, lengths in val_loader:
                token_ids = token_ids.to(device)
                tag_ids = tag_ids.to(device)
                lengths = lengths.to(device)
                max_len = token_ids.size(1)
                mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
                loss = model(token_ids, tag_ids, mask)
                val_loss += loss.item()

        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        print(f"  Epoch {epoch+1}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")

    return train_losses, val_losses


def predict_bilstm_crf(model, data_loader, idx2tag, device=DEVICE):
    """Get predictions from BiLSTM-CRF."""
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for token_ids, tag_ids, lengths in data_loader:
            token_ids = token_ids.to(device)
            lengths = lengths.to(device)
            max_len = token_ids.size(1)
            mask = torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)

            decoded = model.decode(token_ids, mask)

            for i, (seq, length) in enumerate(zip(decoded, lengths)):
                l = length.item()
                pred_tags = [idx2tag[t] for t in seq[:l]]
                true_tags = [idx2tag[tag_ids[i][j].item()] for j in range(l)]
                all_preds.append(pred_tags)
                all_trues.append(true_tags)

    return all_trues, all_preds
