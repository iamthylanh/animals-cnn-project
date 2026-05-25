from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import random

# Mapping tên tiếng Ý → tiếng Anh
italian_to_english = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel"
}

def prepare_dataset():
    raw_path = Path("raw-img")
    
    if not raw_path.exists():
        print("Không tìm thấy thư mục raw-img!")
        return

    print("BẮT ĐẦU CHUẨN BỊ DATASET...\n")

    # 1. Đổi tên thư mục từ tiếng Ý sang tiếng Anh
    print("Đang đổi tên thư mục...")
    for old_name, new_name in italian_to_english.items():
        old_folder = raw_path / old_name
        new_folder = raw_path / new_name
        if old_folder.exists():
            if new_folder.exists():
                shutil.rmtree(new_folder)
            old_folder.rename(new_folder)
            print(f" {old_name} → {new_name}")

    # 2. Xóa file không phải ảnh
    print("\nKiểm tra và xóa file lỗi...")
    error_count = 0
    for folder in raw_path.iterdir():
        if folder.is_dir():
            for file in list(folder.iterdir()):
                if file.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
                    file.unlink()
                    error_count += 1
                    print(f"Xóa: {file.name}")
    print(f"Đã xóa {error_count} file lỗi.")

    # 3. Tạo class_names.txt
    classes = sorted([d.name for d in raw_path.iterdir() if d.is_dir()])
    with open("class_names.txt", "w", encoding="utf-8") as f:
        for name in classes:
            f.write(f"{name}\n")
    print(f"\nĐã tạo class_names.txt ({len(classes)} classes)")

    # 4. Chia dataset thành train / val / test
    print("\nĐang chia train/val/test...")
    train_path = Path("dataset/train")
    val_path = Path("dataset/val")
    test_path = Path("dataset/test")

    for p in [train_path, val_path, test_path]:
        p.mkdir(parents=True, exist_ok=True)

    for class_name in classes:
        class_dir = raw_path / class_name
        images = list(class_dir.glob("*.*"))
        
        # Shuffle
        random.shuffle(images)
        
        # Chia tỷ lệ: 70% train, 15% val, 15% test
        train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

        # Tạo thư mục class trong train/val/test
        (train_path / class_name).mkdir(exist_ok=True)
        (val_path / class_name).mkdir(exist_ok=True)
        (test_path / class_name).mkdir(exist_ok=True)

        # Copy file
        for img in train_imgs:
            shutil.copy(img, train_path / class_name / img.name)
        for img in val_imgs:
            shutil.copy(img, val_path / class_name / img.name)
        for img in test_imgs:
            shutil.copy(img, test_path / class_name / img.name)

        print(f"   📁 {class_name:12} → Train: {len(train_imgs):4} | Val: {len(val_imgs):3} | Test: {len(test_imgs):3}")

    print("\nHOÀN TẤT CHUẨN BỊ DATASET!")
    print(f"Tổng số class: {len(classes)}")
    print("📁 Cấu trúc cuối cùng:")
    print("   - dataset/train/")
    print("   - dataset/val/")
    print("   - dataset/test/")
    print("   - class_names.txt")

if __name__ == "__main__":
    prepare_dataset()