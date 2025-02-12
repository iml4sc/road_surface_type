import argparse
import os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from data.transform import CropTransform
from data.custom_folder import CustomImageFolder
from models.train import train_model
from utils.dir_ops import create_result_dirs, save_images_to_folders
from utils.evaluation import evaluate_model_with_paths
from utils.plotting import plot_multiclass_roc_pr
from utils.metrics import calculate_metrics_from_cm

### ★ Available MODEL LISTS ###
MODEL_LIST = ["inceptionv3", "resnet152v2", "resnet50", "densenet121", "efficientnet_b0"]


def modify_fc_for_model(model_name, model_obj, num_classes):
    """
    Change the last FC (or classifier) layer
    """
    import torch.nn as nn
    if model_name == "inceptionv3":
        model_obj.fc = nn.Linear(model_obj.fc.in_features, num_classes)
    elif model_name in ["resnet152v2", "resnet50"]:
        model_obj.fc = nn.Linear(model_obj.fc.in_features, num_classes)
    elif model_name == "densenet121":
        model_obj.classifier = nn.Linear(model_obj.classifier.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model_obj.classifier[1] = nn.Linear(model_obj.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model_obj


def get_unique_folder_name(base_dir, folder_name):
    """
    >> base_dir/folder_name 
    ex) folder_name = "resnet50_crop"
    -> "results/resnet50_crop" 
        "results/resnet50_crop_1" ,
        "results/resnet50_crop_2", ...
    """
    result_path = os.path.join(base_dir, folder_name)
    if not os.path.exists(result_path):
        return result_path

    i = 1
    while True:
        new_path = f"{result_path}_{i}"
        if not os.path.exists(new_path):
            return new_path
        i += 1


def train_and_validate_one_model(model_name, data_dir, use_crop, epochs, batch_size, lr):
    """
    Train + Validation
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training model={model_name} on device={device}")

    #================================================
    # (A) Model init + input_size
    #================================================
    if model_name == "inceptionv3":
        from torchvision.models import Inception_V3_Weights
        print("  > Inception v3 (aux_logits=True)")
        model_obj = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)
        input_size = 299

    elif model_name == "resnet152v2":
        from torchvision.models import ResNet152_Weights
        print("  > ResNet152 v2")
        model_obj = models.resnet152(weights=ResNet152_Weights.DEFAULT)
        input_size = 224

    elif model_name == "resnet50":
        from torchvision.models import ResNet50_Weights
        print("  > ResNet50")
        model_obj = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        input_size = 224

    elif model_name == "densenet121":
        from torchvision.models import DenseNet121_Weights
        print("  > DenseNet121")
        model_obj = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
        input_size = 224

    elif model_name == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights
        print("  > EfficientNet-B0")
        model_obj = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        input_size = 224

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    #================================================
    # (B) Dataset & DataLoader
    #================================================
    transform_fn = CropTransform((input_size, input_size), use_crop=use_crop)
    common_transforms = transforms.Compose([
        transform_fn,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=common_transforms
    )
    val_dataset = CustomImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=common_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"  > Found {num_classes} classes: {class_names}")

    #================================================
    # (C) Modify FC
    #================================================
    model_obj = modify_fc_for_model(model_name, model_obj, num_classes)
    model_obj.to(device)

    #================================================
    # (D) Train
    #================================================
    print(f"\n  > Training {model_name} ...")
    trained_model = train_model(model_obj, train_loader, device, epochs=epochs, lr=lr)

    #================================================
    # (E) mkdir results/xxx
    #================================================
    base_dir = "results"
    crop_name = "crop" if use_crop else "full"
    sub_folder = f"{model_name}_{crop_name}"
    result_root = get_unique_folder_name(base_dir, sub_folder)
    os.makedirs(result_root, exist_ok=True)
    print(f"  > Result folder: {result_root}")

    # Save the model weights
    model_save_path = os.path.join(result_root, f"{model_name}_model.pth")
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"  > Model saved to: {model_save_path}")

    #================================================
    # (F) Validation
    #     - Inception v3 => aux_logits=False init
    #================================================
    if model_name == "inceptionv3":
        print("  > Re-initializing Inception v3 for Validation (aux_logits=False)")
        eval_model_obj = models.inception_v3(weights=None, aux_logits=False)
        eval_model_obj = modify_fc_for_model(model_name, eval_model_obj, num_classes)
        eval_model_obj.load_state_dict(torch.load(model_save_path), strict=False)
    else:
        if model_name == "resnet152v2":
            eval_model_obj = models.resnet152(weights=None)
        elif model_name == "resnet50":
            eval_model_obj = models.resnet50(weights=None)
        elif model_name == "densenet121":
            eval_model_obj = models.densenet121(weights=None)
        elif model_name == "efficientnet_b0":
            eval_model_obj = models.efficientnet_b0(weights=None)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        eval_model_obj = modify_fc_for_model(model_name, eval_model_obj, num_classes)
        eval_model_obj.load_state_dict(torch.load(model_save_path))

    eval_model_obj.to(device)

    # Data Validation
    val_true, val_pred, val_probs, val_paths = evaluate_model_with_paths(
        eval_model_obj, val_loader, device, num_classes
    )
    val_cm = confusion_matrix(val_true, val_pred)

    ### (1) ROC/PR Curve
    plot_multiclass_roc_pr(val_true, val_probs, class_names, result_root, prefix="val")

    ### (2) Confusion Matrix Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(val_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"[Validation] Confusion Matrix ({model_name})")
    plt.savefig(os.path.join(result_root, f"{model_name}_val_confusion_matrix.png"), dpi=300)
    plt.close()

    ### (3) Classification Report
    val_report = classification_report(val_true, val_pred, target_names=class_names)
    with open(os.path.join(result_root, f"{model_name}_val_classification_report.txt"), "w") as f:
        f.write(val_report)

    ### (4) TP/FP/FN/TN Save (Val)
    val_result_root = os.path.join(result_root, "val_results")
    os.makedirs(val_result_root, exist_ok=True)
    create_result_dirs(val_result_root, class_names)
    save_images_to_folders(val_paths, val_true, val_pred, val_result_root, class_names)

    print(f"\n[Validation Finished for {model_name}] => {result_root}\n")


def main():
    parser = argparse.ArgumentParser(description="Train + Validation for Multiple Models")
    parser.add_argument("--models", type=str, nargs="+", required=True,
                        choices=MODEL_LIST,
                        help="List of models to train (space-separated)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to dataset (with 'train' and 'val' folders)")
    parser.add_argument("--use-crop", action="store_true", default=False,
                        help="Use bottom 1/3 cropping (default=False)")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    ## Model Train // Val --> sequentially
    for model_name in args.models:
        print("==========================================")
        print(f"Start train/val => model: {model_name}")
        print("==========================================")
        train_and_validate_one_model(
            model_name=model_name,
            data_dir=args.data_dir,
            use_crop=args.use_crop,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )

if __name__ == "__main__":
    main()
