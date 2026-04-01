from src.dataset import get_dataloaders
from src.model import get_model
from src.train import train
from src.evaluate import evaluate_full
from src.plots import plot_training, plot_roc

DATA_PATH = "data/dermnet"

train_loader, val_loader, test_loader, classes = get_dataloaders(DATA_PATH)

model = get_model(len(classes))

history = train(model, train_loader, val_loader)

y_true, y_probs = evaluate_full(model, test_loader)

plot_training(history)
plot_roc(y_true, y_probs, len(classes))
