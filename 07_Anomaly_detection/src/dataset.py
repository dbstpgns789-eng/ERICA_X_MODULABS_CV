"""VisA 데이터셋 로딩과 전처리. 노트북 셀 [9]~[11]을 옮긴 것."""
from pathlib import Path

import pandas as pd
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_EXTS = ['.jpg', '.jpeg', '.png', '.bmp']


class VisADataset(Dataset):
    """train_ratio는 정상 사진만 나누는 비율. 불량은 항상 전부 시험 세트로 간다."""

    def __init__(self, data_path, category, transform=None, is_train=True, include_mask=False, train_ratio=0.8):
        self.data_path = Path(data_path)
        self.category = category
        self.transform = transform
        self.is_train = is_train
        self.include_mask = include_mask

        self.image_paths = []
        self.mask_paths = []
        self.labels = []
        self.defect_types = []

        category_path = self.data_path / category
        csv_path = category_path / "image_anno.csv"
        if not csv_path.exists():
            print("CSV 파일이 없습니다. 폴더 구조를 기반으로 데이터를 로드합니다.")
            self._load_from_folder_structure()
            return

        df = pd.read_csv(csv_path)
        print(f"CSV 컬럼: {df.columns.tolist()}")
        normal_df = df[df['label'] == 'normal'].copy()
        anomaly_df = df[df['label'] != 'normal'].copy()

        if self.is_train:
            train_size = int(len(normal_df) * train_ratio)
            selected_df = normal_df.iloc[:train_size]
            print(f"훈련용 정상 데이터: {len(selected_df)}개")
        else:
            test_normal_size = len(normal_df) - int(len(normal_df) * train_ratio)
            test_normal_df = normal_df.iloc[-test_normal_size:] if test_normal_size > 0 else pd.DataFrame()
            selected_df = pd.concat([test_normal_df, anomaly_df], ignore_index=True)
            print(f"테스트용 정상 데이터: {len(test_normal_df)}개")
            print(f"테스트용 비정상 데이터: {len(anomaly_df)}개")

        for _, row in selected_df.iterrows():
            image_path = self._resolve(row['image'])
            if not image_path.exists():
                print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
                continue
            self.image_paths.append(str(image_path))
            self.labels.append(0 if row['label'] == 'normal' else 1)
            self.defect_types.append(row['label'])

            if pd.notna(row['mask']) and self.include_mask:
                mask_path = self._resolve(row['mask'])
                self.mask_paths.append(str(mask_path) if mask_path.exists() else None)
            else:
                self.mask_paths.append(None)

    def _resolve(self, rel_path):
        # CSV 경로가 'pcb4/Data/...'처럼 카테고리로 시작하는 경우가 있어 중복을 피한다
        if rel_path.startswith(self.category + '/'):
            return self.data_path / rel_path
        return self.data_path / self.category / rel_path

    def _load_from_folder_structure(self):
        images_path = self.data_path / self.category / "Data" / "Images"
        masks_path = self.data_path / self.category / "Data" / "Masks"

        for img_path in (images_path / "Normal").glob("*"):
            if img_path.suffix.lower() in IMG_EXTS:
                self.image_paths.append(str(img_path))
                self.labels.append(0)
                self.defect_types.append('normal')
                self.mask_paths.append(None)

        if self.is_train:
            return

        for img_path in (images_path / "Anomaly").glob("*"):
            if img_path.suffix.lower() in IMG_EXTS:
                self.image_paths.append(str(img_path))
                self.labels.append(1)
                self.defect_types.append('anomaly')
                if self.include_mask:
                    mask_path = masks_path / "Anomaly" / (img_path.stem + '.png')
                    self.mask_paths.append(str(mask_path) if mask_path.exists() else None)
                else:
                    self.mask_paths.append(None)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        label = self.labels[idx]

        mask = None
        if self.include_mask and self.mask_paths[idx] is not None:
            try:
                mask = Image.open(self.mask_paths[idx]).convert('L')
            except Exception:
                mask = None

        if self.transform:
            image = self.transform(image)
            if mask is not None:
                mask = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])(mask)

        result = {'image': image, 'label': label, 'defect_type': self.defect_types[idx]}
        if mask is not None:
            result['mask'] = mask
        return result


def get_visa_transforms(flip=False):
    # 실험 4·9: 좌우 뒤집기는 정상 범위를 넓혀 불량을 숨겼다 (PaDiM 0.974→0.958, SVM 0.872→0.817)
    train_steps = [transforms.Resize((256, 256))]
    if flip:
        train_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    train_steps += [transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)]

    train_transform = transforms.Compose(train_steps)
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    return train_transform, test_transform


def create_visa_data_loaders(data_path, category, batch_size=32, include_mask=False, train_ratio=0.8, flip=False):
    train_transform, test_transform = get_visa_transforms(flip)
    train_dataset = VisADataset(data_path, category, transform=train_transform, is_train=True,
                                include_mask=include_mask, train_ratio=train_ratio)
    test_dataset = VisADataset(data_path, category, transform=test_transform, is_train=False,
                               include_mask=include_mask, train_ratio=train_ratio)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, test_loader


if __name__ == "__main__":
    import sys
    data = sys.argv[1] if len(sys.argv) > 1 else "./VisA"
    category = sys.argv[2] if len(sys.argv) > 2 else "pcb4"
    tr, te = create_visa_data_loaders(data, category, batch_size=16)
    print(f"train {len(tr.dataset)}장 / test {len(te.dataset)}장 (불량 {sum(te.dataset.labels)}장)")
