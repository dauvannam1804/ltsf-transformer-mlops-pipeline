import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model_normalized(model, loader, scaler, device='cpu'):
    model.eval()
    preds, trues = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)

            for i in range(len(out)):
                pred_raw = scaler.inverse_transform(out[i].cpu().numpy().reshape(-1, 1)).flatten()
                true_raw = scaler.inverse_transform(y[i].cpu().numpy().reshape(-1, 1)).flatten()
                preds.extend(pred_raw)
                trues.extend(true_raw)
    
    mse = mean_squared_error(trues, preds)
    mae = mean_absolute_error(trues, preds)
    rmse = np.sqrt(mse)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "r2": r2_score(trues, preds),
        "predictions": np.array(preds),
        "actuals": np.array(trues)
    }
