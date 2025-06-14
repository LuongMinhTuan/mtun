import re
import unicodedata

def normalize(text):
    text = unicodedata.normalize("NFD", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return text.strip()

def load_pairs(path="data_train.txt"):
    with open(path, encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
    
    # Không kiểm tra lỗi nữa, import thẳng
    pairs = [line.split("|", 1) for line in lines]
    return [(normalize(q), normalize(a)) for q, a in pairs]
