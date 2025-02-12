import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def evaluate_model_with_paths(model, data_loader, device, num_classes):
    """
    Model validation
    - y_true, y_pred, softmax output, img path
    """
    model.eval()
    y_true_list = []
    y_pred_list = []
    probs_list = []
    paths_list = []

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for images, labels, paths in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            ## For Inception v3 (aux_logits)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits

            # Prediction prop
            probs = softmax(outputs)
            _, preds = torch.max(outputs, 1)

            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            paths_list.extend(paths)

    all_probs = np.concatenate(probs_list, axis=0)
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    return y_true, y_pred, all_probs, paths_list
