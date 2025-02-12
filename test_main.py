import argparse
import os
import re
import torch
import torchvision.models as models
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from data.transform import CropTransform
from data.custom_folder import CustomImageFolder
from utils.dir_ops import create_result_dirs, save_images_to_folders
from utils.evaluation import evaluate_model_with_paths
from utils.plotting import plot_multiclass_roc_pr
from utils.metrics import calculate_metrics_from_cm

def guess_model_name_from_pth(pth_filename):
    """
    Extract Model Name
    """
    match = re.match(r"^([a-z0-9_]+)_model\.pth$", pth_filename)
    if match:
        return match.group(1) 
    return None


def modify_fc_for_model(model_name, model_obj, num_classes):
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


def test_model(weights_path, data_dir, use_crop, batch_size):
    """
    Model Testing
        - weights_path: *.pth
        - data_dir: test
        - use_crop: True -->  bottom 1/3 crop
        - batch_size: test batch size
    """
    ## 1) .pth
    pth_filename = os.path.basename(weights_path)        # "resnet50_model.pth"
    pth_folder = os.path.dirname(weights_path)           # "./results/resnet50_crop"
    model_name = guess_model_name_from_pth(pth_filename) # "resnet50"
    if model_name is None:
        raise ValueError(f"Cannot infer model name from: {pth_filename}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Testing model={model_name} with weights={weights_path}")
    print(f"       on device={device}")

    ## 2) Model structure init (weights=None)
    if model_name == "inceptionv3":
        print("  > Re-initializing Inception v3 (aux_logits=False)")
        model_obj = models.inception_v3(weights=None, aux_logits=False)
        input_size = 299
    elif model_name == "resnet152v2":
        model_obj = models.resnet152(weights=None)
        input_size = 224
    elif model_name == "resnet50":
        model_obj = models.resnet50(weights=None)
        input_size = 224
    elif model_name == "densenet121":
        model_obj = models.densenet121(weights=None)
        input_size = 224
    elif model_name == "efficientnet_b0":
        model_obj = models.efficientnet_b0(weights=None)
        input_size = 224
    else:
        raise ValueError(f"Unsupported or unknown model_name: {model_name}")

    ## 3) Test Dataset load + # of classes
    transform_fn = CropTransform((input_size, input_size), use_crop=use_crop)
    import torchvision.transforms as transforms
    common_transforms = transforms.Compose([
        transform_fn,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])

    test_dataset = CustomImageFolder(root=os.path.join(data_dir, "test"),
                                    transform=common_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=4)
    class_names = test_dataset.classes
    num_classes = len(class_names)
    print(f"  > Detected {num_classes} classes in test set: {class_names}")

    ## 4) FC modify
    model_obj = modify_fc_for_model(model_name, model_obj, num_classes)

    ## 5) load weights
    model_obj.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model_obj.to(device)

    ## 6) evaluate
    from utils.evaluation import evaluate_model_with_paths
    test_true, test_pred, test_probs, test_paths = evaluate_model_with_paths(
        model_obj, test_loader, device, num_classes
    )
    test_cm = confusion_matrix(test_true, test_pred)

    ## 7) save results
    result_root = pth_folder 
    print(f"  > Test results will be saved under: {result_root}")

    ### (A) ROC/PR Curve
    plot_multiclass_roc_pr(test_true, test_probs, class_names, result_root, prefix="test")

    ### (B) Confusion Matrix Visualization
    plt.figure(figsize=(10, 8))
    sns.heatmap(test_cm, annot=True, fmt='d', cmap='Oranges',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"[Test] Confusion Matrix ({model_name})")
    plt.savefig(os.path.join(result_root, f"{model_name}_test_confusion_matrix.png"), dpi=300)
    plt.close()

    ### (C) Classification Report
    test_report = classification_report(test_true, test_pred, target_names=class_names)
    with open(os.path.join(result_root, f"{model_name}_test_classification_report.txt"), "w") as f:
        f.write(test_report)

    ### (D) TP/FP/FN/TN save
    test_result_root = os.path.join(result_root, "test_results")
    os.makedirs(test_result_root, exist_ok=True)
    create_result_dirs(test_result_root, class_names)
    save_images_to_folders(test_paths, test_true, test_pred, test_result_root, class_names)

    ### (E) Detailed metrics
    from utils.metrics import calculate_metrics_from_cm
    metrics_dict = calculate_metrics_from_cm(test_cm, class_names)
    print(f"\n[Test] Detailed Metrics for {model_name}:")
    for cls_name, vals in metrics_dict.items():
        print(f"  [{cls_name}] TP={vals['TP']}, FP={vals['FP']}, "
            f"FN={vals['FN']}, TN={vals['TN']}, "
            f"Precision={vals['Precision']:.4f}, Recall={vals['Recall']:.4f}, F1={vals['F1-score']:.4f}")

    metric_save_path = os.path.join(result_root, f"{model_name}_test_metrics.txt")
    with open(metric_save_path, "w") as f:
        for cls_name, vals in metrics_dict.items():
            f.write(f"{cls_name} => TP={vals['TP']}, FP={vals['FP']}, FN={vals['FN']}, TN={vals['TN']}, "
                    f"Precision={vals['Precision']:.4f}, Recall={vals['Recall']:.4f}, "
                    f"F1={vals['F1-score']:.4f}\n")

    print(f"\n[TEST COMPLETED for {model_name}] See results in: {result_root}\n")


def main():
    parser = argparse.ArgumentParser(description="Test a single model with .pth path")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to trained .pth file (e.g. './results/resnet50_crop/resnet50_model.pth')")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to dataset (with 'test' folder)")
    parser.add_argument("--use-crop", action="store_true", default=False,
                        help="Use bottom 1/3 cropping (default=False)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for test")

    args = parser.parse_args()
    test_model(weights_path=args.weights,
            data_dir=args.data_dir,
            use_crop=args.use_crop,
            batch_size=args.batch_size)

if __name__ == "__main__":
    main()
