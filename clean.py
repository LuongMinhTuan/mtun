# clean_data.py

def clean_file(input_path="data.txt", clean_path="clean_data.txt", error_path="error_lines.txt"):
    with open(input_path, encoding="utf-8") as f:
        lines = f.read().strip().split("\n")

    clean_lines = []
    error_lines = []

    for i, line in enumerate(lines):
        if "|" in line:
            parts = line.split("|", 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                clean_lines.append(line)
            else:
                error_lines.append(f"{i+1}: {line}")
        else:
            error_lines.append(f"{i+1}: {line}")

    with open(clean_path, "w", encoding="utf-8") as f:
        f.write("\n".join(clean_lines))

    with open(error_path, "w", encoding="utf-8") as f:
        f.write("\n".join(error_lines))

    print(f"✅ Đã lọc xong: {len(clean_lines)} dòng hợp lệ, {len(error_lines)} dòng lỗi.")

# Chạy luôn nếu gọi script
if __name__ == "__main__":
    clean_file()
