import torch
import torch.nn as nn
import random
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import gc
import torch.optim.lr_scheduler as lr_scheduler
import time



seed = 42

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['TORCH_USE_CUDA_DSA'] = '1'


class RadarDatasetChunks(Dataset):
    def __init__(self, data_dir, chunk_prefix="train", total_chunks=30):
        self.data_dir = data_dir
        self.chunk_prefix = chunk_prefix
        self.chunk_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"{chunk_prefix}_radar_chunk")])
        self.label_files = sorted([f for f in os.listdir(data_dir) if f.startswith(f"{chunk_prefix}_labels_chunk")])
        assert len(self.chunk_files) == len(self.label_files), "Mismatch between radar and label chunk files"

        # Precompute sizes for each chunk
        self.chunk_sizes = [np.load(os.path.join(data_dir, f)).shape[0] for f in self.chunk_files]
        self.total_samples = sum(self.chunk_sizes)
        self.current_chunk_idx = 0
        self.load_chunk(self.current_chunk_idx)

        self.label_mapping = {
            'LayOnFloor': 0,
            'LayOnSofa': 1,
            'SitOnFloor': 2,
            'Sitting': 3,
            'Standing': 4,
            'Walking': 5,
            'MoveSitOnChair': 6,
            'MoveStandingPickupFromFloor': 7,
            'MoveStandingPickupFromTable': 8
        }

    def load_chunk(self, chunk_idx):
        radar_file = os.path.join(self.data_dir, self.chunk_files[chunk_idx])
        labels_file = os.path.join(self.data_dir, self.label_files[chunk_idx])

        self.current_data = np.load(radar_file)
        self.current_labels = np.load(labels_file)
        self.current_chunk_size = self.current_data.shape[0]

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        cumulative_size = 0
        for chunk_idx, chunk_size in enumerate(self.chunk_sizes):
            if cumulative_size + chunk_size > idx:
                if chunk_idx != self.current_chunk_idx:
                    self.current_chunk_idx = chunk_idx
                    self.load_chunk(chunk_idx)

                # Calculate the sample index within the current chunk
                sample_idx = idx - cumulative_size

                # Convert the radar data and labels to torch tensors
                radar_sample = torch.tensor(self.current_data[sample_idx], dtype=torch.float32)

                # Convert label (string) to integer using the label mapping
                label_sample_str = self.current_labels[sample_idx]
                if isinstance(label_sample_str, str) or isinstance(label_sample_str, np.str_):
                    label_sample = self.label_mapping[label_sample_str]  # Map string to integer
                else:
                    label_sample = label_sample_str  # It is Already numeric

                label_sample = torch.tensor(label_sample, dtype=torch.long)

                return radar_sample, label_sample

            cumulative_size += chunk_size

def create_data_loader(data_dir, batch_size, chunk_prefix="train", total_chunks=30):
    dataset = RadarDatasetChunks(data_dir=data_dir, chunk_prefix=chunk_prefix, total_chunks=total_chunks)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(chunk_prefix == "train"), num_workers=8, pin_memory=True, prefetch_factor=2)
    return loader



# Updated train function with additional features
def train_model3(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir, scheduler, early_stop_patience=10):
    train_history = []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_val_acc = 0
    patience_counter = 0

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    history_file = os.path.join(save_dir, 'training_history.txt')
    check_file = os.path.join(save_dir, 'check.txt')

    with open(history_file, 'w') as f:
        f.write("Epoch,Train_Loss,Train_Acc,Val_Loss,Val_Acc\n")

    with open(check_file, 'w') as f:
        f.write("Training started...\n")

    # scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    for epoch in range(num_epochs):
        print(f"Device in train_model2 {device}")

        # START Training phase
        model.train()
        train_loss, train_correct = 0, 0
        total_train = 0
        start_training_time= time.time()
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit="batch") as pbar:
            for step, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                # Accumulate training metrics
                train_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                train_correct += (preds == labels).sum().item()
                total_train += labels.size(0)

                pbar.update(1)  # Update the progress bar

                # Every 250 steps, log and save intermediate results
                if (step + 1) % 250 == 0:
                    average_train_loss = train_loss / total_train
                    average_train_acc = 100 * train_correct / total_train
                    print(f"Step [{step+1}], Train Loss: {average_train_loss:.4f}, Train Acc: {average_train_acc:.4f}")

                    if average_train_loss < best_train_loss * 0.95:
                        best_train_loss = average_train_loss
                        print("Best train loss achieved, saving model.")
                        torch.save(model.state_dict(), os.path.join(save_dir, f'best_train_model_epoch_{epoch+1}.pth'))

        # Validation phase
        model.eval()
        val_loss, val_correct = 0, 0
        total_val = 0

        with torch.no_grad():
            with tqdm(total=len(val_loader), desc=f"Epoch {epoch+1}/{num_epochs} [Val]", unit="batch") as pbar_val:
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * inputs.size(0)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    total_val += labels.size(0)

                    pbar_val.update(1)  # Update the progress bar

        average_train_loss = train_loss / total_train
        average_train_acc = 100 * train_correct / total_train
        average_val_loss = val_loss / total_val
        average_val_acc = 100 * val_correct / total_val

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {average_train_loss:.4f}, Train Acc: {average_train_acc:.4f}, Val Loss: {average_val_loss:.4f}, Val Acc: {average_val_acc:.4f}")

        with open(history_file, 'a') as f:
            f.write(f"{epoch + 1},{average_train_loss:.4f},{average_train_acc:.4f},{average_val_loss:.4f},{average_val_acc:.4f}\n")

        if average_val_loss < best_val_loss * 0.95:
            best_val_loss = average_val_loss
            patience_counter = 0
            print("Best validation loss achieved, saving model.")
            torch.save(model.state_dict(), os.path.join(save_dir, f'best_val_model_epoch_{epoch+1}.pth'))
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"Early stopping at epoch {epoch + 1}. Validation loss did not improve for {early_stop_patience} epochs.")
            break

        scheduler.step()

        gc.collect()
        torch.cuda.empty_cache()

    return model, train_history