
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from monai.utils import set_determinism
from monai.losses import DiceLoss, DiceCELoss
from monai.data import DataLoader, Dataset
import torch
import matplotlib.pyplot as plt
import glob
from data_transforms import get_transforms
from model import AttentionUNET, DeeperAttentionUNET, deep_supervision_loss
from infernece import validation
from utils import load_checkpoint, save_checkpoint, get_files
from CONFIG import NET_ARGS


def main(train_files, val_files, batch_size, load, cp_path, max_epochs, loss_curve_path, add_ce_loss=False, deeper_net=False):

    set_determinism(seed=0) # set the random seed for reproducabilty
    train_transforms = get_transforms(train=True)
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_transforms = get_transforms(train=False)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeeperAttentionUNET(**NET_ARGS) if deeper_net else AttentionUNET(**NET_ARGS)
    model.to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True) if add_ce_loss else DiceLoss(to_onehot_y=True, softmax=True) 
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    print(loss_function)
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    metric_values = []

    if load:
        model, optimizer, last_epoch, best_metric = load_checkpoint(cp_path, model, optimizer)
        epoch = last_epoch + 1

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = (
                batch_data["image"].to(device),
                batch_data["label"].to(device),
            )
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = deep_supervision_loss(outputs, labels, loss_function)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"{step}/{len(train_ds) // train_loader.batch_size}, " f"train_loss: {loss.item():.4f}")
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        plt.plot(epoch_loss_values)
        plt.xlabel('epochs')
        plt.ylabel('Training Loss')
        plt.title('Training Loss Curve')
        plt.show()
        plt.savefig(loss_curve_path)

        if (epoch + 1) % val_interval == 0:
            metric, _, _ , _ = validation(model, val_loader, device)
            metric_values.append(metric)
            if metric[1] > best_metric:
                best_metric = metric[1]
                best_metric_epoch = epoch + 1
                save_checkpoint(cp_path, model, optimizer, epoch, metric, epoch_loss_values)
                print("saved new best metric model")
            print(
                f"current epoch: {epoch + 1} current mean dice: {metric}"
                f"\nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

if __name__ == "__main__":
    root_dir = ""
    data_dir = os.path.join(root_dir, "data")
    train_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
    experiment_dir = os.path.join(root_dir, "conv_mapping")
    if not os.path.exists(experiment_dir):
        os.mkdir(experiment_dir)
    loss_curve_path = os.path.join(experiment_dir, "loss.png")
    data_split_path = os.path.join(experiment_dir, "data_split.json")
    cp_path = os.path.join(experiment_dir, "best_metric_model.pth")
    load = False
    batch_size = 2
    max_epochs = 600
    train_files, val_files, _ = get_files(data_split_path, train_images, train_labels)
    main(train_files, val_files, batch_size, load, cp_path, max_epochs, loss_curve_path, add_ce_loss=False, deeper_net=False)