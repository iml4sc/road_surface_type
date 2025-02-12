import numpy as np

def calculate_metrics_from_cm(cm, class_names):
    """
    Confusion Matrix ---> TP, FP, FN, TN, Precision, Recall, F1-score
    """
    num_classes = cm.shape[0]
    metrics = {}

    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        metrics[class_names[i]] = {
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
            'Precision': precision,
            'Recall': recall,
            'F1-score': f1
        }
    return metrics
