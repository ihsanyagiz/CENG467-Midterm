"""
Task 5: LSTM Language Model
"""

import torch
import torch.nn as nn
import math
from tqdm import tqdm

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEVICE, TASK5_BPTT


class LSTMLanguageModel(nn.Module):
    """LSTM-based language model."""
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Tie weights
        if embed_dim == hidden_dim:
            self.fc.weight = self.embedding.weight

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        embedded = self.dropout(self.embedding(x))
        output, hidden = self.lstm(embedded, hidden)
        output = self.dropout(output)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size, device=DEVICE):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))


def train_lstm_lm(model, train_data, val_data, epochs, lr, bptt=TASK5_BPTT, device=DEVICE):
    """Train the LSTM language model."""
    model = model.to(device)
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        hidden = model.init_hidden(train_data.size(0), device)

        for i in range(0, train_data.size(1) - 1, bptt):
            data, targets = get_batch_from_source(train_data, i, bptt)
            if data.size(1) == 0:
                continue

            # Detach hidden state
            hidden = tuple(h.detach() for h in hidden)

            optimizer.zero_grad()
            output, hidden = model(data, hidden)
            loss = criterion(output.reshape(-1, output.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_train = total_loss / max(num_batches, 1)
        train_losses.append(avg_train)

        # Validation
        val_loss = evaluate_lm(model, val_data, criterion, bptt, device)
        val_losses.append(val_loss)

        train_ppl = math.exp(min(avg_train, 20))
        val_ppl = math.exp(min(val_loss, 20))
        print(f"  Epoch {epoch+1}: Train Loss={avg_train:.4f} (PPL={train_ppl:.2f}), Val Loss={val_loss:.4f} (PPL={val_ppl:.2f})")

    return train_losses, val_losses


def evaluate_lm(model, data, criterion, bptt=TASK5_BPTT, device=DEVICE):
    """Evaluate the language model and return average loss."""
    model.eval()
    total_loss = 0
    num_batches = 0
    hidden = model.init_hidden(data.size(0), device)

    with torch.no_grad():
        for i in range(0, data.size(1) - 1, bptt):
            x, targets = get_batch_from_source(data, i, bptt)
            if x.size(1) == 0:
                continue
            hidden = tuple(h.detach() for h in hidden)
            output, hidden = model(x, hidden)
            loss = criterion(output.reshape(-1, output.size(-1)), targets.reshape(-1))
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def get_batch_from_source(source, i, bptt):
    """Get a batch from source tensor."""
    seq_len = min(bptt, source.size(1) - 1 - i)
    data = source[:, i:i + seq_len]
    target = source[:, i + 1:i + 1 + seq_len]
    return data, target


def generate_text(model, vocab, seed_text="the", max_len=100, temperature=0.8, device=DEVICE):
    """Generate text from the LSTM language model."""
    model.eval()
    idx2word = {v: k for k, v in vocab.items()}

    tokens = seed_text.lower().split()
    input_ids = [vocab.get(t, vocab["<unk>"]) for t in tokens]

    hidden = model.init_hidden(1, device)
    generated = list(tokens)

    with torch.no_grad():
        # Feed seed tokens
        for token_id in input_ids[:-1]:
            inp = torch.LongTensor([[token_id]]).to(device)
            _, hidden = model(inp, hidden)

        # Generate
        current_id = input_ids[-1]
        for _ in range(max_len):
            inp = torch.LongTensor([[current_id]]).to(device)
            output, hidden = model(inp, hidden)

            # Apply temperature
            logits = output[0, -1] / temperature
            probs = torch.softmax(logits, dim=0)
            next_id = torch.multinomial(probs, 1).item()

            if next_id == vocab["<eos>"]:
                break

            word = idx2word.get(next_id, "<unk>")
            generated.append(word)
            current_id = next_id

    return " ".join(generated)
