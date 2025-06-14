# split_5k_rest.py

def split_5k(input_path="clean_data.txt", train_path="data_train.txt", rest_path="data_rest.txt"):
    with open(input_path, encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    train_lines = lines[:5000]
    rest_lines = lines[5000:]

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines))

    with open(rest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rest_lines))

    print(f"✅ Đã tách:")
    print(f"   ➤ {len(train_lines)} dòng trong {train_path}")
    print(f"   ➤ {len(rest_lines)} dòng trong {rest_path}")

if __name__ == "__main__":
    split_5k()
