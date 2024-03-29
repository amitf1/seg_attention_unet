import os
import torch
import json
from datetime import datetime
from sklearn.model_selection import train_test_split


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

def load_checkpoint(cp_path, model, optimizer=None):
    checkpoint = torch.load(cp_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    last_epoch = checkpoint['epoch']
    best_metric = checkpoint['best_metric']
 
    return model, optimizer, last_epoch, best_metric

def get_files(data_split_path, train_images, train_labels):

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
    return train_files, val_files, test_files
