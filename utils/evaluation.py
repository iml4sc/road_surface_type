# MIT License
#
# Copyright (c) 2025 Integrated Media Systems Center (IMSC),
# University of Southern California
# Jooyoung Yoo
# Dr. Seon Ho Kim
#
# This file is licensed under the MIT License. See the LICENSE file for details.

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

    with torch.no_grad():
        for images, labels, paths in tqdm(data_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            ## For Inception v3 (aux_logits)
            if hasattr(outputs, 'logits'):
                outputs = outputs.logits

            ## Apply activation function based on classification type
            if num_classes == 2 and outputs.shape[1] == 1:
                ##> Binary classification: use sigmoid and create two probability columns
                probs = torch.sigmoid(outputs)
                probs = torch.cat([1 - probs, probs], dim=1)
            else:
                ##> Multi-class classification: apply softmax
                probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)
            
            y_true_list.extend(labels.cpu().numpy())
            y_pred_list.extend(preds.cpu().numpy())
            probs_list.append(probs.cpu().numpy())
            paths_list.extend(paths)

    all_probs = np.concatenate(probs_list, axis=0)
    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    return y_true, y_pred, all_probs, paths_list
