from sklearn.metrics import classification_report, roc_auc_score
import torch
import numpy as np

def evaluate_full(model, loader):
    device = next(model.parameters()).device
    model.eval()

    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            outputs = model(x)

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            y_true.extend(y.numpy())
            y_pred.extend(preds)
            y_probs.extend(probs)

    print(classification_report(y_true, y_pred))

    auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    print("AUC:", auc)

    return y_true, y_probs
