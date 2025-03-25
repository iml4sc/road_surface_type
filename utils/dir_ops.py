# MIT License
#
# Copyright (c) 2025 Integrated Media Systems Center (IMSC),
# University of Southern California
# Jooyoung Yoo
# Dr. Seon Ho Kim
#
# This file is licensed under the MIT License. See the LICENSE file for details.

import os
import cv2

def create_result_dirs(result_root, class_names):
    """
    Create folder : TP, FP, TN, FN
    """
    for cls in class_names:
        for cat in ['TP', 'FP', 'TN', 'FN']:
            os.makedirs(os.path.join(result_root, cls, cat), exist_ok=True)


def save_images_to_folders(images_paths, y_true, y_pred, result_root, class_names):
    """
    TP, FP, FN, TN Image save
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_thickness = 2
    gt_color = (0, 0, 255)    ## RED
    pred_color = (0, 255, 0)  ## GREEN

    for idx, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        img_path = images_paths[idx]
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        ## GT, Pred labels on the image
        gt_text = f"GT: {class_names[true_label]}"
        pred_text = f"Pred: {class_names[pred_label]}"
        cv2.putText(img, gt_text, (10, 30), font, font_scale, gt_color, font_thickness, cv2.LINE_AA)
        cv2.putText(img, pred_text, (10, 60), font, font_scale, pred_color, font_thickness, cv2.LINE_AA)

        ## Save TP, FP, FN, TN
        for cls_idx, cls_name in enumerate(class_names):
            if true_label == cls_idx and pred_label == cls_idx:
                target_folder = os.path.join(result_root, cls_name, 'TP')
            elif true_label != cls_idx and pred_label == cls_idx:
                target_folder = os.path.join(result_root, cls_name, 'FP')
            elif true_label == cls_idx and pred_label != cls_idx:
                target_folder = os.path.join(result_root, cls_name, 'FN')
            else:
                target_folder = os.path.join(result_root, cls_name, 'TN')

            save_path = os.path.join(target_folder, img_name)
            cv2.imwrite(save_path, img)
