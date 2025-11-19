import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from typing import Union

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device: Union[str, torch.device] = torch.device("cpu")):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    
    model.to(device)
    train_losses, val_losses = [], []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            epoch_train += loss.item()

        train_losses.append(epoch_train / len(train_loader))
        
        model.eval()
        epoch_val = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                epoch_val += criterion(model(x), y).item()
        val_losses.append(epoch_val / len(val_loader))

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs} | Train {train_losses[-1]:.4f} | Val {val_losses[-1]:.4f}")
    
    return model, train_losses, val_losses
