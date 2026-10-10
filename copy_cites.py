import os
import shutil

# Путь к списку
list_file = "C:/Proj_edge_distr/filtered_networks.txt"
# Куда копировать
dest_dir = "C:/Proj_edge_distr/selected/"
os.makedirs(dest_dir, exist_ok=True)

with open(list_file, "r", encoding="utf-8") as f:
    paths = [line.strip() for line in f if line.strip()]

copied, missing = 0, 0
for src in paths:
    if not os.path.exists(src):
        print(f"❌ Нет файла: {src}")
        missing += 1
        continue

    filename = os.path.basename(src)
    dst = os.path.join(dest_dir, filename)

    # Если в разных подпапках могут быть файлы с одинаковыми именами —
    # добавь префикс из хэша пути, чтобы не перезаписать:
    # import hashlib
    # h = hashlib.md5(src.encode()).hexdigest()[:6]
    # dst = os.path.join(dest_dir, f"{h}_{filename}")

    shutil.copy2(src, dst)
    copied += 1

print(f"\n✅ Скопировано: {copied}")
print(f"❌ Пропущено (нет файла): {missing}")