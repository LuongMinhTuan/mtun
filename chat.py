import torch
import torch.nn as nn
import torch.nn.functional as F

# Load word2idx và idx2word
word2idx = torch.load("word2idx.pth")
idx2word = torch.load("idx2word.pth")
vocab_size = len(word2idx)

# Hàm mã hóa và giải mã
def encode(text, max_len=10):
    ids = [word2idx.get(w, 0) for w in text.lower().split()]
    return ids[:max_len] + [0]*(max_len - len(ids))

def decode(ids):
    words = [idx2word[i] for i in ids if i != 0 and idx2word[i] not in ["<sos>", "<eos>"]]
    return " ".join(words)

# Mô hình giống như train.py
class Chatbot(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.encoder = nn.LSTM(32, 64, batch_first=True)
        self.decoder = nn.LSTM(32, 64, batch_first=True)
        self.out = nn.Linear(64, vocab_size)

    def forward(self, x, y, hidden=None):
        x_embed = self.embed(x)
        _, (h, c) = self.encoder(x_embed)

        y_embed = self.embed(y)
        out, _ = self.decoder(y_embed, (h, c))
        return self.out(out)

# Load model đã huấn luyện
model = Chatbot()
model.load_state_dict(torch.load("mtun.pth"))
model.eval()

# Chat loop
print("🤖 Chatbot sẵn sàng! Gõ 'exit' để thoát.")
while True:
    inp = input("Nguoi: ")
    if inp.lower() == "exit":
        break

    x = torch.tensor([encode(inp)], dtype=torch.long)
    y = torch.tensor([[word2idx["<sos>"]]], dtype=torch.long)
    response = []

    with torch.no_grad():
        hidden = None
        for _ in range(10):  # Tối đa 10 từ
            output = model(x, y, hidden)
            last_word_logits = output[0, -1]
            next_word = torch.argmax(F.softmax(last_word_logits, dim=-1)).item()
            if idx2word[next_word] == "<eos>":
                break
            response.append(next_word)
            y = torch.cat([y, torch.tensor([[next_word]])], dim=1)

    print("Bot:", decode(response))
