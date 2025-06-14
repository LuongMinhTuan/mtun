import torch
import torch.nn as nn
from prepare import load_pairs

pairs = load_pairs()
input_texts = [p[0] for p in pairs]
target_texts = ["<sos> " + p[1] + " <eos>" for p in pairs]

# Tạo vocab
all_text = " ".join(input_texts + target_texts).split()
vocab = sorted(set(all_text))
word2idx = {w: i+1 for i, w in enumerate(vocab)}  # start từ 1
word2idx["<pad>"] = 0
idx2word = {i: w for w, i in word2idx.items()}
vocab_size = len(word2idx)

def encode(text, max_len=10):
    ids = [word2idx.get(w, 0) for w in text.split()]
    return ids[:max_len] + [0]*(max_len - len(ids))

X = torch.tensor([encode(q) for q in input_texts])
Y = torch.tensor([encode(a) for a in target_texts])

class Chatbot(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.encoder = nn.LSTM(32, 64, batch_first=True)
        self.decoder = nn.LSTM(32, 64, batch_first=True)
        self.out = nn.Linear(64, vocab_size)

    def forward(self, x, y):
        x_embed = self.embed(x)
        _, (h, c) = self.encoder(x_embed)

        y_embed = self.embed(y)
        out, _ = self.decoder(y_embed, (h, c))
        return self.out(out)

model = Chatbot()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 200
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X, Y[:, :-1])
    loss = criterion(output.reshape(-1, vocab_size), Y[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()

    print(f"✅ Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")  # 👈 Đếm từng epoch

# Lưu model và từ điển
torch.save(model.state_dict(), "mtun.pth")
torch.save(word2idx, "word2idx.pth")
torch.save(idx2word, "idx2word.pth")
