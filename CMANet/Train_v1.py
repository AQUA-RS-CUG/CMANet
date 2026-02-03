import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import transforms
from sklearn.model_selection import train_test_split
import os
import time
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
from Configs.Config_for_Train_v1 import args
from model.CMANet_model import CMANet
from User_Dataset import CMANet_Dataset
from Focal_Loss import MultiFocalLoss
from osgeo import gdal

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['PROJ_LIB'] = r'D:\APP\Anaconda\envs\sdsd_torch\Lib\site-packages\osgeo\data\proj'
os.environ['GDAL_DATA'] = r'D:\APP\Anaconda\envs\sdsd_torch\Lib\site-packages\osgeo\data'
gdal.UseExceptions()


__foldername__ = ['Flat', 'Sand', 'Submerged', 'Vegetation', 'Water']  # Category Name
optical_train_sample_path_list = [os.path.join(args.optical_path, item) for item in
                                  os.listdir(args.optical_path)]
sar_train_sample_path_list = [os.path.join(args.sar_path, item) for item in
                              os.listdir(args.sar_path)]
optical_train_image_path_list = []
sar_train_image_path_list = []
train_label_list = []
index = 0
for i in range(len(__foldername__)):
    optical_image_path_list = [os.path.join(optical_train_sample_path_list[i], item) for item in
                               os.listdir(optical_train_sample_path_list[i])]
    sar_image_path_list = [os.path.join(sar_train_sample_path_list[i], item) for item in
                           os.listdir(sar_train_sample_path_list[i])]
    optical_train_image_path_list.extend(optical_image_path_list)
    sar_train_image_path_list.extend(sar_image_path_list)

    for j in range(len(optical_image_path_list)):
        train_label_list.append(index)
    index += 1

train_sample = list(zip(optical_train_image_path_list, sar_train_image_path_list, train_label_list))
random.shuffle(train_sample)
optical_train_image_path_list[:], sar_train_image_path_list[:], train_label_list[:] = zip(*train_sample)

# train_optical_paths, val_optical_paths, train_sar_paths, val_sar_paths, train_labels, val_labels = train_test_split(
#     optical_train_image_path_list,
#     sar_train_image_path_list,
#     train_label_list,
#     test_size=0.2,
#     random_state=42,
#     stratify=train_label_list
# )

train_dataset = CMANet_Dataset(optical_train_image_path_list, sar_train_image_path_list, train_label_list)
# train_dataset = CMANet_Dataset(train_optical_paths, train_sar_paths, train_labels)
# val_dataset = CMANet_Dataset(val_optical_paths, val_sar_paths, val_labels)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True, drop_last=True)
# val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CMANet(in_channel1=12, in_channel2=4, num_class=args.num_class_new, hidden_dim=1024, num_heads=8, dropout=0.1)
model.to(device)
criterion_level = MultiFocalLoss(num_class=args.num_class_new).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)

all_train_loss_list = []
val_loss_list = []
s_train_loss_list = []
o_train_loss_list = []
epoch_list = []

lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5, T_mult=4, eta_min=1e-6)
print('>>>>>>>>>>>>>>>  CMANet Model Training Start  <<<<<<<<<<<<<<<')
train_start_time = time.time()
best_loss = float('inf')
best_epoch = 0
patience = 10
no_improve_epochs = 0


def format_time(seconds):
    hour = seconds // 3600
    minute = (seconds % 3600) // 60
    second = seconds % 60
    return f"{int(hour)}小时{int(minute)}分钟{int(second)}秒"


for epoch in range(1, args.epoch + 1):
    start_time = time.time()
    model.train()
    train_loss = 0
    # all_avg_loss = 0
    print("Epoch {}/{}".format(epoch, args.epoch))
    print("learning rate: {}".format(optimizer.param_groups[0]["lr"]))

    for i, batch in tqdm(enumerate(train_dataloader)):
        optical_train_batch = batch['optical_image'].to(device)
        sar_train_batch = batch['sar_image'].to(device)
        label_train_batch = batch['label'].to(device).long()
        optimizer.zero_grad()
        output_level = model(optical_train_batch, sar_train_batch)
        loss = criterion_level(output_level, label_train_batch)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    if epoch < 25:
        lr_scheduler.step()

    train_loss /= len(train_dataloader)
    all_train_loss_list.append(train_loss)

    # model.eval()
    # val_loss = 0
    # with torch.no_grad():
    #     for batch in val_dataloader:
    #         optical_val_batch = batch['optical_image'].to(device)
    #         sar_val_batch = batch['sar_image'].to(device)
    #         label_val_batch = batch['label'].to(device).long()
    #         output_val = model(optical_val_batch, sar_val_batch)
    #         loss = criterion_level(output_val, label_val_batch)
            # val_loss += loss.item()
    # val_loss /= len(val_dataloader)
    # val_loss_list.append(val_loss)
    epoch_list.append(epoch)
    print(f"Epoch {epoch}/{args.epoch}, Train Loss: {train_loss:.4f}")
    # print(f"Epoch {epoch}/{args.epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    # if val_loss < best_loss:
    #     best_loss = val_loss
    #     best_epoch = epoch
    #     no_improve_epochs = 0
    # else:
    #     no_improve_epochs += 1

    # if no_improve_epochs >= patience:
    #     print(f"Validation loss has not improved for {patience} epochs. Stopping training.")
    #     break

    print('========================== CMANet Model Training Info ==========================')
    run_time = time.time() - start_time
    print(f'Epoch {epoch} Training Time：{format_time(run_time)}\n')

train_end_time = time.time()
all_run_time = train_end_time - train_start_time
print(f'Total time the training program：{format_time(all_run_time)}')
print("Lowest loss is epoch {}, The value is {:.4f}".format(best_epoch, best_loss))

plt.figure(figsize=(10, 6))
plt.rcParams['font.family'] = 'Times New Roman'
plt.plot(epoch_list, all_train_loss_list, marker='o', linestyle='-', color='b', label='All Train Loss', markersize=3)
# plt.plot(epoch_list, val_loss_list, marker='s', linestyle='--', color='r', label='Val Loss', markersize=3)
plt.xlabel('Epoch', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.title('Training Loss Over Epochs', fontsize=18)
plt.grid(True)
plt.legend(loc='best', fontsize=16)
plt.savefig(args.model_save_path + r'\CMANet_V1.png', dpi=200)
plt.show()

torch.save(model.state_dict(), args.model_save_path + r'\CMANet_V001.pth')
