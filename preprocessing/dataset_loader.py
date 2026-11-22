import os
import glob
from typing import List, Tuple, Dict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import RAW_DATA_DIR, RANDOM_SEED, HELD_OUT_GENERATOR, VAL_SPLIT, TEST_SPLIT, IMAGE_SIZE

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)

class ImageDataset(Dataset):
    def __init__(self, file_paths: List[str], labels: List[int], transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            # Handle corrupt images by returning a blank image (or ideally filter them beforehand)
            image = Image.new('RGB', IMAGE_SIZE)
            
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

def discover_dataset() -> Tuple[Dict[str, List[str]], Dict[str, List[str]], List[str]]:
    """
    Discovers the dataset in RAW_DATA_DIR.
    Expected structure:
      data/raw/REAL/{source}/*.jpg
      data/raw/AI-GENERATED/{generator}/*.jpg
      
    Returns:
      real_files: Dict grouping REAL files by source
      ai_files: Dict grouping AI-GENERATED files by generator
      errors: List of any issues found
    """
    real_dir = os.path.join(RAW_DATA_DIR, "REAL")
    ai_dir = os.path.join(RAW_DATA_DIR, "AI-GENERATED")
    
    real_files = {}
    ai_files = {}
    errors = []
    
    # Check REAL
    if not os.path.exists(real_dir):
        errors.append(f"Directory not found: {real_dir}")
    else:
        for source in os.listdir(real_dir):
            source_path = os.path.join(real_dir, source)
            if os.path.isdir(source_path):
                files = glob.glob(os.path.join(source_path, "*.*"))
                valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                real_files[source] = valid_files

    # Check AI-GENERATED
    if not os.path.exists(ai_dir):
        errors.append(f"Directory not found: {ai_dir}")
    else:
        for gen in os.listdir(ai_dir):
            gen_path = os.path.join(ai_dir, gen)
            if os.path.isdir(gen_path):
                files = glob.glob(os.path.join(gen_path, "*.*"))
                valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                ai_files[gen] = valid_files
                
    return real_files, ai_files, errors

def create_splits(real_files: Dict[str, List[str]], ai_files: Dict[str, List[str]], max_train_samples: int = None, max_eval_samples: int = None):
    """
    Creates train/val/test/held_out splits.
    Held out generator is completely isolated from train/val.
    Optional max_train_samples / max_eval_samples allow controlled subsampling for rapid training/benchmarks.
    """
    set_seed()
    
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []
    held_out_paths, held_out_labels = [], []
    
    # Process REAL
    all_real = []
    for source, files in real_files.items():
        all_real.extend(files)
        
    if all_real:
        r_train_val, r_test = train_test_split(all_real, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
        r_train, r_val = train_test_split(r_train_val, test_size=VAL_SPLIT/(1-TEST_SPLIT), random_state=RANDOM_SEED)

        # Carve disjoint REAL images for held-out eval so they never overlap ID test/train.
        n_held_real = min(250, max(1, len(r_train) // 8))
        if n_held_real > 0 and len(r_train) > n_held_real + 50:
            r_train, r_held_real = train_test_split(r_train, test_size=n_held_real, random_state=RANDOM_SEED)
            held_out_paths.extend(r_held_real)
            held_out_labels.extend([0] * len(r_held_real))
    else:
        r_train, r_val, r_test = [], [], []
        
    # Process AI-GENERATED
    all_ai_train_val = []
    for gen, files in ai_files.items():
        if gen == HELD_OUT_GENERATOR:
            held_out_paths.extend(files)
            held_out_labels.extend([1]*len(files))
        else:
            all_ai_train_val.extend(files)
            
    if all_ai_train_val:
        a_train_val, a_test = train_test_split(all_ai_train_val, test_size=TEST_SPLIT, random_state=RANDOM_SEED)
        a_train, a_val = train_test_split(a_train_val, test_size=VAL_SPLIT/(1-TEST_SPLIT), random_state=RANDOM_SEED)
    else:
        a_train, a_val, a_test = [], [], []

    # Subsampling if requested (preserving class balance)
    if max_train_samples is not None and max_train_samples > 0:
        n_r = min(len(r_train), max_train_samples // 2)
        n_a = min(len(a_train), max_train_samples - n_r)
        r_train = np.random.choice(r_train, size=n_r, replace=False).tolist()
        a_train = np.random.choice(a_train, size=n_a, replace=False).tolist()
        
    if max_eval_samples is not None and max_eval_samples > 0:
        n_val_r = min(len(r_val), max_eval_samples // 2)
        n_val_a = min(len(a_val), max_eval_samples - n_val_r)
        r_val = np.random.choice(r_val, size=n_val_r, replace=False).tolist()
        a_val = np.random.choice(a_val, size=n_val_a, replace=False).tolist()

        n_test_r = min(len(r_test), max_eval_samples // 2)
        n_test_a = min(len(a_test), max_eval_samples - n_test_r)
        r_test = np.random.choice(r_test, size=n_test_r, replace=False).tolist()
        a_test = np.random.choice(a_test, size=n_test_a, replace=False).tolist()
        
        # Subsample held-out as well
        h_idx_0 = [i for i, l in enumerate(held_out_labels) if l == 0]
        h_idx_1 = [i for i, l in enumerate(held_out_labels) if l == 1]
        n_h0 = min(len(h_idx_0), max_eval_samples // 2)
        n_h1 = min(len(h_idx_1), max_eval_samples - n_h0)
        sel_0 = np.random.choice(h_idx_0, size=n_h0, replace=False).tolist()
        sel_1 = np.random.choice(h_idx_1, size=n_h1, replace=False).tolist()
        sel_all = sel_0 + sel_1
        held_out_paths = [held_out_paths[i] for i in sel_all]
        held_out_labels = [held_out_labels[i] for i in sel_all]

    train_paths.extend(r_train); train_labels.extend([0]*len(r_train))
    val_paths.extend(r_val); val_labels.extend([0]*len(r_val))
    test_paths.extend(r_test); test_labels.extend([0]*len(r_test))

    train_paths.extend(a_train); train_labels.extend([1]*len(a_train))
    val_paths.extend(a_val); val_labels.extend([1]*len(a_val))
    test_paths.extend(a_test); test_labels.extend([1]*len(a_test))
        
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), (held_out_paths, held_out_labels)

def get_dataloaders(batch_size=32, max_train_samples: int = None, max_eval_samples: int = None):
    real_files, ai_files, errors = discover_dataset()
    
    train_data, val_data, test_data, held_out_data = create_splits(
        real_files, ai_files, max_train_samples=max_train_samples, max_eval_samples=max_eval_samples
    )
    
    transform_train = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pin = torch.cuda.is_available()
    workers = 2 if pin else 0
    train_loader = DataLoader(ImageDataset(*train_data, transform=transform_train), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin)
    val_loader = DataLoader(ImageDataset(*val_data, transform=transform_test), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    test_loader = DataLoader(ImageDataset(*test_data, transform=transform_test), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    held_out_loader = DataLoader(ImageDataset(*held_out_data, transform=transform_test), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    
    return train_loader, val_loader, test_loader, held_out_loader

import io
from PIL import ImageFilter

class RobustnessAugmentation:
    """
    Online data augmentation during training to instill resilience
    against JPEG compression, slight blur, and contrast changes.
    """
    def __init__(self, p_jpeg=0.3, p_blur=0.2):
        self.p_jpeg = p_jpeg
        self.p_blur = p_blur

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p_jpeg:
            q = int(np.random.randint(55, 90))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=q)
            buf.seek(0)
            img = Image.open(buf).convert('RGB')

        if np.random.rand() < self.p_blur:
            img = img.filter(ImageFilter.GaussianBlur(radius=float(np.random.uniform(0.5, 1.5))))

        return img

class DualModalDataset(Dataset):
    """
    Dataset that yields both spatial image tensor and 2D FFT multi-band frequency spectrum tensor.
    """
    def __init__(self, file_paths: List[str], labels: List[int], spatial_transform=None, augment=False):
        self.file_paths = file_paths
        self.labels = labels
        self.spatial_transform = spatial_transform
        self.augmenter = RobustnessAugmentation() if augment else None
        from preprocessing.frequency_transform import compute_multi_band_fft, fft_spectrum_to_tensor
        self.compute_multi_band_fft = compute_multi_band_fft
        self.fft_spectrum_to_tensor = fft_spectrum_to_tensor

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception:
            image = Image.new('RGB', IMAGE_SIZE)

        if self.augmenter:
            image = self.augmenter(image)

        # Spatial tensor
        if self.spatial_transform:
            spatial_tensor = self.spatial_transform(image)
        else:
            spatial_tensor = transforms.ToTensor()(image)

        # Multi-band frequency tensor [3, H, W]
        multi_band = self.compute_multi_band_fft(image, size=IMAGE_SIZE)
        fft_tensor = self.fft_spectrum_to_tensor(multi_band)

        return spatial_tensor, fft_tensor, torch.tensor(label, dtype=torch.long)

def get_dual_dataloaders(batch_size=32, max_train_samples: int = None, max_eval_samples: int = None, augment: bool = True):
    real_files, ai_files, errors = discover_dataset()
    
    train_data, val_data, test_data, held_out_data = create_splits(
        real_files, ai_files, max_train_samples=max_train_samples, max_eval_samples=max_eval_samples
    )
    
    transform_train = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    pin = torch.cuda.is_available()
    workers = 2 if pin else 0
    train_loader = DataLoader(DualModalDataset(*train_data, spatial_transform=transform_train, augment=augment), batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=pin)
    val_loader = DataLoader(DualModalDataset(*val_data, spatial_transform=transform_test, augment=False), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    test_loader = DataLoader(DualModalDataset(*test_data, spatial_transform=transform_test, augment=False), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    held_out_loader = DataLoader(DualModalDataset(*held_out_data, spatial_transform=transform_test, augment=False), batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    
    return train_loader, val_loader, test_loader, held_out_loader
