import torch
import torch.nn as nn
from utils import *
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_arch.ResNet18SeparableSingleView import ResNet18SeparableSingleView
from model_arch.ResNet18SeparableSingleViewAttention import ResNet18SeparableSingleViewAttention
from model_arch.ViT16Custom import ViT16Custom
from model_arch.SingleViewViT import SingleViewViT

MODELS_MAP={
    ResNet18SeparableSingleView: "ResNet18SeparableSingleView",
    ResNet18SeparableSingleViewAttention: "ResNet18SeparableSingleViewAttention"
}

# batch_size = 512
batch_size = 64
num_epochs = 200
# learning_rate = 1e-3
# Small learning rate for vision Transformer
learning_rate = 1e-4
lr_str = "{:.0e}".format(learning_rate)
device = torch.device("cuda")
print(device)


criterion = nn.CrossEntropyLoss()
# model = ResNet18SeparableSingleView(name='single_view_experiment', num_classes=9).to(device)
# model = ResNet18SeparableSingleViewAttention(name='single_view_experiment', num_classes=9).to(device)
# model = ViT16Custom(name='ViT16Custom_h12_d12', num_classes=9).to(device)
model = SingleViewViT(name='SingleViewViT', num_classes=9).to(device)


# model_str = MODELS_MAP[model]
model_str = model.get_name()

print(f"model_str: {model_str}")

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
if isinstance(optimizer, torch.optim.Adam):
    optimizer_str="Adam"
    print("Adam")
else:
    optimizer_str="Other"

# Gradient Clipping (ViT)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
# scheduler = CosineAnnealingLR(optimizer, T_max=50)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
if isinstance(scheduler,torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    print("STUFF")
    scheduler_str = "CosineAnnealingWarmRestarts"
elif isinstance(scheduler, CosineAnnealingLR):
    scheduler_str = "CosineAnnealingLR"
elif isinstance(scheduler, lr_scheduler.StepLR):
    scheduler_str = "StepLR"
elif isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
    scheduler_str = "ReduceLROnPlateau"
else:
    print("Unknown")
    scheduler_str = "UK"

def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

model.apply(initialize_weights)

data_dir = r"/mnt/datasets_nfs/h3trinh/"
save_dir = f"/mnt/slurm_nfs/h3trinh/MV_Radar_Based_HAR/{model_str}/scheduler_{scheduler_str}/lr_{lr_str[-1]}_b{batch_size}_{optimizer_str}"
print("Creating train loader..")
train_loader = create_data_loader(data_dir, batch_size=batch_size, chunk_prefix="train", total_chunks=10)
print("Finish creating train loader")

print("Creating val loader.. ")
val_loader = create_data_loader(data_dir, batch_size=batch_size, chunk_prefix="val", total_chunks=1)
print("Finish creating val loader")

print(f"Model is on device: {next(model.parameters()).device}")

# Train the model with early stopping
device = torch.device("cuda")
print(f"save_dir: {save_dir}")

# trained_model, history = train_model2(
trained_model, history = train_model3(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_epochs,
    device=device,
    save_dir=save_dir,
    scheduler=scheduler,
    early_stop_patience=15  # Stop if validation loss doesn't improve for 5 epochs
)
