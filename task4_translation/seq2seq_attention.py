"""
Task 4: Seq2Seq with Bahdanau Attention
Encoder-Decoder with GRU and additive attention for EN→DE translation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, TASK4_TEACHER_FORCING


class TranslationDataset(Dataset):
    def __init__(self, src, tgt, src_lens, tgt_lens):
        self.src = torch.LongTensor(src)
        self.tgt = torch.LongTensor(tgt)
        self.src_lens = torch.LongTensor(src_lens)
        self.tgt_lens = torch.LongTensor(tgt_lens)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx], self.src_lens[idx], self.tgt_lens[idx]


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, hidden = self.rnn(embedded)
        # hidden: [2, batch, hidden] -> combine forward/backward
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1)))
        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W_enc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_dec = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: [batch, hidden]
        # encoder_outputs: [batch, src_len, hidden*2]
        dec = self.W_dec(decoder_hidden).unsqueeze(1)  # [batch, 1, hidden]
        enc = self.W_enc(encoder_outputs)               # [batch, src_len, hidden]
        energy = torch.tanh(dec + enc)                   # [batch, src_len, hidden]
        attention = self.v(energy).squeeze(2)            # [batch, src_len]
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(hidden_dim)
        self.rnn = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim + hidden_dim * 2 + embed_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden, encoder_outputs):
        # input_token: [batch, 1]
        embedded = self.dropout(self.embedding(input_token))  # [batch, 1, embed]

        attn_weights = self.attention(hidden, encoder_outputs)  # [batch, src_len]
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hidden*2]

        rnn_input = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))
        hidden = hidden.squeeze(0)

        prediction = self.fc_out(torch.cat((output.squeeze(1), context.squeeze(1), embedded.squeeze(1)), dim=1))
        return prediction, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=TASK4_TEACHER_FORCING):
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        # First input: <sos>
        input_token = tgt[:, 0].unsqueeze(1)

        for t in range(1, tgt_len):
            prediction, hidden, _ = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t] = prediction

            if random.random() < teacher_forcing_ratio:
                input_token = tgt[:, t].unsqueeze(1)
            else:
                input_token = prediction.argmax(1).unsqueeze(1)

        return outputs


def train_seq2seq(model, train_loader, val_loader, epochs, lr, tgt_pad_idx, device=DEVICE):
    """Train the Seq2Seq model."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_pad_idx)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for src, tgt, src_lens, tgt_lens in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False):
            src, tgt = src.to(device), tgt.to(device)
            optimizer.zero_grad()
            output = model(src, tgt)
            # output: [batch, tgt_len, vocab], tgt: [batch, tgt_len]
            output = output[:, 1:].reshape(-1, output.size(-1))
            tgt = tgt[:, 1:].reshape(-1)
            loss = criterion(output, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / len(train_loader)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, tgt, src_lens, tgt_lens in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src, tgt, teacher_forcing_ratio=0)
                output = output[:, 1:].reshape(-1, output.size(-1))
                tgt_flat = tgt[:, 1:].reshape(-1)
                loss = criterion(output, tgt_flat)
                val_loss += loss.item()
        avg_val = val_loss / len(val_loader)
        val_losses.append(avg_val)
        print(f"  Epoch {epoch+1}: Train Loss={avg_train:.4f}, Val Loss={avg_val:.4f}")

    return train_losses, val_losses


def translate_seq2seq(model, src_sentence, src_vocab, tgt_vocab, max_len=50, device=DEVICE):
    """Translate a single sentence using the Seq2Seq model (greedy decoding)."""
    model.eval()
    idx2word = {v: k for k, v in tgt_vocab.items()}

    # Encode source
    from preprocess import encode_sentence
    src_ids, _ = encode_sentence(src_sentence, src_vocab, max_len)
    src_tensor = torch.LongTensor([src_ids]).to(device)

    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
        input_token = torch.LongTensor([[tgt_vocab["<sos>"]]]).to(device)

        output_tokens = []
        for _ in range(max_len):
            prediction, hidden, _ = model.decoder(input_token, hidden, encoder_outputs)
            top1 = prediction.argmax(1).item()

            if top1 == tgt_vocab["<eos>"]:
                break
            output_tokens.append(idx2word.get(top1, "<unk>"))
            input_token = torch.LongTensor([[top1]]).to(device)

    return " ".join(output_tokens)
