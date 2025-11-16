import matplotlib.pyplot as plt

def plot_loss_curves(loss_history, horizons):
    for hz in horizons:
        plt.figure(figsize=(10, 6))
        for model in loss_history:
            if hz in loss_history[model]:
                plt.plot(loss_history[model][hz]["train"], label=f"{model} Train")
                plt.plot(loss_history[model][hz]["val"], "--", label=f"{model} Val")
        plt.legend()
        plt.title(f"Loss curves ({hz})")
        plt.show()
