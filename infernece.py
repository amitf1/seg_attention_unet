import glob
import json
import os
import torch
from monai.metrics import DiceMetric, SurfaceDistanceMetric
from monai.inferers import sliding_window_inference
from monai.data import DataLoader, Dataset, decollate_batch
from monai.transforms import (
    AsDiscrete,
    Compose,
    Activations,
    SaveImage

)
from data_transforms import get_transforms
from utils import load_checkpoint
from CONFIG import ROI_SIZE, NET_ARGS
from model import AttentionUNET, DeeperAttentionUNET

def calculate_precision_recall(pred_label, true_label, label=1):

    pred_label = torch.stack(pred_label)[:, label, ...]
    true_label = torch.stack(true_label)[:, label, ...]

    # True positives, false positives, and false negatives
    TP = (pred_label * true_label).sum()
    FP = ((pred_label == 1) & (true_label == 0)).sum()
    FN = ((pred_label == 0) & (true_label == 1)).sum()
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    return precision.item(), recall.item()

def validation(model, val_loader, device, save_images_path=None):
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
    surface_distance = SurfaceDistanceMetric(include_background=True, reduction="mean_batch", symmetric=True)
    # applying final activation to the output and creating one_hot segmentation for calculating metrics
    post_pred = Compose([Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=2)]) 
    # creating one_hot segmentation from the ground truth for calculating metrics
    post_label = Compose([AsDiscrete(to_onehot=2)])

    if save_images_path is not None:
        # if saving images - reverse one hot by argmax to go back to one channel segmentation map
        save_pred = Compose([AsDiscrete(argmax=True), SaveImage(output_dir=save_images_path, output_postfix="pred", separate_folder=False)])
        save_label = Compose([AsDiscrete(argmax=True), SaveImage(output_dir=save_images_path, output_postfix="label", separate_folder=False)])
        save_image = SaveImage(output_dir=save_images_path, output_postfix="image", separate_folder=False)

    model.eval()

    precision, recall = [], []
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            
            sw_batch_size = 4
            # run model inference using sliding windows over patches of the real image, aggregating the results with gaussian weight and 0.25 overalp (for smoothness) to a full image
            val_outputs = sliding_window_inference(val_inputs, ROI_SIZE, sw_batch_size, model, overlap=0.25, mode="gaussian") 
            decol_outputs = decollate_batch(val_outputs) # 1 tensor to a list of batch tensors
            decol_labels = decollate_batch(val_labels)
            val_outputs = [post_pred(i) for i in decol_outputs] # apply activations and one hot encoding
            val_labels = [post_label(i) for i in decol_labels]
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)
            surface_distance(y_pred=val_outputs, y=val_labels)
            p, r = calculate_precision_recall(val_outputs, val_labels) # calculate precision and recall on the positive label
            precision.append(p)
            recall.append(r)
            if save_images_path is not None:
                decol_inputs = decollate_batch(val_inputs)
                for i in range(len(decol_inputs)):
                    save_image(decol_inputs[i])
                    save_label(val_labels[i]) 
                    save_pred(val_outputs[i])
        mean_precision, mean_recall = torch.mean(torch.tensor(precision)), torch.mean(torch.tensor(recall))
        # aggregate the final result - output is the mean over the batches for each channel, 0 is the background 
        mean_dice = dice_metric.aggregate()
        mean_surface_distance = surface_distance.aggregate()
        # reset the status for next validation round
        dice_metric.reset()
        surface_distance.reset()
    return mean_dice, mean_surface_distance, mean_precision, mean_recall

def inference(data_split_path, cp_path, save_images_path=None, deeper_net=False):
    with open(data_split_path) as json_file:
        data_split = json.load(json_file)
        test_files = data_split["test_files"]

    val_transforms = get_transforms(train=False)
    test_ds = Dataset(data=test_files, transform=val_transforms)
    test_loader = DataLoader(test_ds, batch_size=1)
    model = DeeperAttentionUNET(**NET_ARGS) if deeper_net else AttentionUNET(**NET_ARGS)

    model, _, last_epoch, best_dice = load_checkpoint(cp_path, model) # ignore optimizer in inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mean_dice, mean_surface_distance, mean_precision, mean_recall = validation(model, test_loader, device=device,save_images_path=save_images_path)
    label_dice_test = mean_dice[1] # get only the label metric, ignore background - label 0
    label_sd_test = mean_surface_distance[1]
    best_dice_val = best_dice[1]

    print(
            f"weights from training epoch: {last_epoch + 1}" # print epoch number with numbering starting from 1
            f"\nmean test dice: {label_dice_test:.4f} "
            f"\nmean test surface distance: {label_sd_test:.4f} "
            f"\nmean test precision: {mean_precision:.4f} "
            f"\nmean test recall: {mean_recall:.4f} "
            f"\nmean validation dice: {best_dice_val:.4f} "

        )
    
if __name__ == "__main__":
    root_dir = ""
    data_dir = os.path.join(root_dir, "data")
    train_images = sorted(glob.glob(os.path.join(data_dir, "images", "*.nii.gz")))
    train_labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.nii.gz")))
    experiment_dir = "baseline"
    data_split_path = os.path.join(root_dir, experiment_dir, "data_split.json")
    cp_path = os.path.join(root_dir, experiment_dir, "best_metric_model.pth")
    save_images_path = os.path.join(root_dir, experiment_dir, "outputs")
    if not os.path.exists(save_images_path):
        os.mkdir(save_images_path)
    inference(data_split_path, cp_path, save_images_path=None)