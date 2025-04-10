# MIT License
#
# Copyright (c) 2025 Integrated Media Systems Center (IMSC),
# University of Southern California
# Jooyoung Yoo
# Dr. Seon Ho Kim
#
# This file is licensed under the MIT License. See the LICENSE file for details.

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize

def plot_multiclass_roc_pr(y_true, probs, class_names, result_root, prefix="val"):
    """
    Binary/Multi-calss ROC / PR curve
    """
    num_classes = len(class_names)
    
    probs = np.asarray(probs)
    if probs.ndim == 1:
        probs = np.expand_dims(probs, axis=1)
    if num_classes == 2 and probs.shape[1] < 2:
        probs = np.concatenate([1 - probs, probs], axis=1)

    ### label_binarize
    y_true_bin = label_binarize(y_true, classes=range(num_classes))

    if num_classes == 2 and y_true_bin.shape[1] == 1:
        ## Binary-class
        #================================================
        # 1) ROC Curves
        #================================================
        y_true_bin = np.concatenate([1 - y_true_bin, y_true_bin], axis=1)
        fpr, tpr, _ = roc_curve(y_true_bin[:, 1], probs[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"{class_names[1]} (AUC={roc_auc:.2f})")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.title(f"Binary ROC ({prefix})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(result_root, f"{prefix}_roc_curve.png"), dpi=300)
        plt.close()

        #================================================
        # 2) Precision-Recall Curves
        #================================================
        prec, rec, _ = precision_recall_curve(y_true_bin[:, 1], probs[:, 1])
        pr_auc = auc(rec, prec)
        plt.figure(figsize=(8, 6))
        plt.plot(rec, prec, label=f"{class_names[1]} (AUC={pr_auc:.2f})")
        plt.title(f"Binary Precision-Recall ({prefix})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(result_root, f"{prefix}_pr_curve.png"), dpi=300)
        plt.close()

    else:
        ## Multi-class
        #================================================
        # 1) ROC Curves
        #================================================
        fpr_dict, tpr_dict, roc_auc_dict = {}, {}, {}
        plt.figure(figsize=(8, 6))
        for i in range(num_classes):
            fpr_dict[i], tpr_dict[i], _ = roc_curve(y_true_bin[:, i], probs[:, i])
            roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
            plt.plot(fpr_dict[i], tpr_dict[i], label=f"{class_names[i]} (AUC={roc_auc_dict[i]:.2f})")

        ## macro-average
        all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(num_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(num_classes):
            mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
        mean_tpr /= num_classes
        macro_auc = auc(all_fpr, mean_tpr)
        plt.plot(all_fpr, mean_tpr, color='navy', linestyle='--',
                label=f"macro-average (AUC={macro_auc:.2f})")
        plt.title(f"Multiclass ROC ({prefix})")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(result_root, f"{prefix}_roc_curve.png"), dpi=300)
        plt.close()

        #================================================
        # 2) Precision-Recall Curves
        #================================================
        prec_dict, rec_dict, pr_auc_dict = {}, {}, {}
        plt.figure(figsize=(8, 6))
        for i in range(num_classes):
            prec_dict[i], rec_dict[i], _ = precision_recall_curve(y_true_bin[:, i], probs[:, i])
            pr_auc_dict[i] = auc(rec_dict[i], prec_dict[i])
            plt.plot(rec_dict[i], prec_dict[i], label=f"{class_names[i]} (AUC={pr_auc_dict[i]:.2f})")

        # macro-average (Simple version)
        all_rec = np.unique(np.concatenate([rec_dict[i] for i in range(num_classes)]))
        mean_prec = np.zeros_like(all_rec)
        for i in range(num_classes):
            mean_prec += np.interp(all_rec, rec_dict[i][::-1], prec_dict[i][::-1])[::-1]
        mean_prec /= num_classes
        macro_pr_auc = auc(all_rec, mean_prec)
        plt.plot(all_rec, mean_prec, color='navy', linestyle='--',
                label=f"macro-average (AUC={macro_pr_auc:.2f})")
        plt.title(f"Multiclass Precision-Recall ({prefix})")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend(loc="lower left")
        plt.savefig(os.path.join(result_root, f"{prefix}_pr_curve.png"), dpi=300)
        plt.close()
