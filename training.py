
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from monai.utils import set_determinism
from monai.metrics import DiceMetric
from monai.losses import DiceLoss, DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
from monai.config import print_config
import torch
import matplotlib.pyplot as plt
import glob
from data_transforms import get_transforms
from sklearn.model_selection import train_test_split
import json
from datetime import datetime
from model import AttentionUNET, deep_supervision_loss
from monai.transforms import (
    AsDiscrete,
    Compose,
    Activations
)


def save_checkpoint(cp_path, model, optimizer, last_epoch, best_metric, epoch_loss):
    checkpoint = {
        'time': datetime.now().strftime("%y_%m_%d_%H_%M_%S"),
        'epoch': last_epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'epoch_loss': epoch_loss,
        'model_state_dict': model.state_dict(),

        }

    torch.save(checkpoint, cp_path)

def load_checkpoint(cp_path, model, optimizer):
    checkpoint = torch.load(cp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    best_metric = checkpoint['best_metric']
 
    return model, optimizer, last_epoch, best_metric

def main(train_files, val_files, batch_size, load, cp_path, max_epochs, loss_curve_path, add_ce_loss=False):

    set_determinism(seed=0) # set the random seed for reproducabilty
    train_transforms = get_transforms(train=True)
    train_ds = Dataset(data=train_files, transform=train_transforms)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_transforms = get_transforms(train=False)
    val_ds = Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionUNET(in_channels=1, out_channels=2, n_deep_supervision=3)
    model.to(device)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True) if add_ce_loss else DiceLoss(to_onehot_y=True, softmax=True) 
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
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
            model.eval()
            with torch.no_grad():
                for val_data in val_loader:
                    val_inputs, val_labels = (
                        val_data["image"].to(device),
                        val_data["label"].to(device),
                    )
                    roi_size = (96, 96, 96)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, overlap=0.25, mode="gaussian")
            
                    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)])
                    post_label = Compose([AsDiscrete(to_onehot=2)])
                    val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                    val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)

                # aggregate the final mean dice result
                metric = dice_metric.aggregate()
                # reset the status for next validation round
                dice_metric.reset()

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
    root_dir = "/nvcr/algo/projects/workdir/afeldman/pancreas"
    data_dir = os.path.join(root_dir, "data")
    train_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
    loss_curve_path = os.path.join(root_dir, "loss.png")
    data_split_path = os.path.join(root_dir, "data_split.json")

    cp_path = os.path.join(root_dir, "best_metric_model.pth")
    load = False

    batch_size = 2
    max_epochs = 600
    if not os.path.exists(data_split_path):
        print("creating data split")
        data_dicts = [{"image": image_name, "label": label_name} for image_name, label_name in zip(train_images, train_labels)]
        train_files, test_files = train_test_split(data_dicts, train_size=0.75, random_state=0)
        train_files, val_files = train_test_split(train_files, train_size=0.85, random_state=0)
        data_split = {"train_files" : train_files, "val_files" : val_files, "test_files" : test_files}
        with open(data_split_path, "w") as outfile: 
            json.dump(data_split, outfile)
    else:
        print(f"loading data split from {data_split_path}")
        with open(data_split_path) as json_file:
            data_split = json.load(json_file)
        train_files, val_files, test_files = data_split["train_files"], data_split["val_files"], data_split["test_files"]
    main(train_files, val_files, batch_size, load, cp_path, max_epochs, loss_curve_path, add_ce_loss=True)