import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, device, epochs=20, lr=1e-4):
    """
    Let's train the Model 
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{epochs}")

            for images, labels in tepoch:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()

                outputs = model(images)
                ## For Inception v3(aux_logits)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                tepoch.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Accuracy': f"{(100.0 * correct / total):.2f}%"
                })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        print(f"[Epoch {epoch+1}/{epochs}] Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    return model
