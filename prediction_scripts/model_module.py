"""
Module for loading a specific PyTorch model and running inference.
Automatically detect if the model is Inception v3 (then use input_size=299).
"""

import torch
import torch.nn.functional as F
import torchvision.models as models

# 3 classes for road surface
CLASS_NAMES = ["Asphalt", "Paved", "Unpaved"]

def guess_model_name_from_path(model_path):
    """
    Simple utility to guess the model type from the file name or path.
    e.g., "inceptionv3_model.pth" -> 'inceptionv3'
          "resnet152v2_model.pth" -> 'resnet152v2'
          ...
    """
    lower_path = model_path.lower()
    if "inceptionv3" in lower_path:
        return "inceptionv3"
    elif "resnet152v2" in lower_path:
        return "resnet152v2"
    elif "resnet50" in lower_path:
        return "resnet50"
    elif "efficientnet_b0" in lower_path:
        return "efficientnet_b0"
    elif "densenet121" in lower_path:
        return "densenet121"
    else:
        return None

def load_model(model_path, device):
    """
    Load a PyTorch model from a .pth file, automatically detect the model architecture.
    Modify the final layer for len(CLASS_NAMES) classes.
    """
    model_name = guess_model_name_from_path(model_path)
    if model_name is None:
        raise ValueError(f"Cannot detect model type from path: {model_path}")

    if model_name == "inceptionv3":
        # Inception v3
        model = models.inception_v3(weights=None, aux_logits=False)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    elif model_name == "resnet152v2":
        model = models.resnet152(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))

    elif model_name == "densenet121":
        model = models.densenet121(weights=None)
        model.classifier = torch.nn.Linear(model.classifier.in_features, len(CLASS_NAMES))
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint, strict=False)

    model.to(device)
    model.eval()

    return model, model_name


def get_input_size_for_model(model_name):
    """
    If the model is 'inceptionv3', return 299.
    Otherwise, return 224 by default.
    """
    if model_name == "inceptionv3":
        return 299
    else:
        return 224


def predict_road_surface(model, img_tensor, device):
    """
    Run a single-image inference and return (label, confidence).
    """
    img_tensor = img_tensor.unsqueeze(0).to(device)  # shape (1, C, H, W)
    with torch.no_grad():
        outputs = model(img_tensor)
        if hasattr(outputs, "logits"):
            outputs = outputs.logits  # In case of Inception
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1)
        pred_label = CLASS_NAMES[pred_idx.item()]
        confidence = probs[0, pred_idx.item()].item()
    return pred_label, confidence
