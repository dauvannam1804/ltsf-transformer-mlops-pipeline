def plot_loss_curves(loss_history, horizons, save_dir=None):
    import matplotlib.pyplot as plt
    import os

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for hz in horizons:
        plt.figure(figsize=(10, 6))
        for model in loss_history:
            if hz in loss_history[model]:
                plt.plot(loss_history[model][hz]["train"], label=f"{model} Train")
                plt.plot(loss_history[model][hz]["val"], "--", label=f"{model} Val")
        plt.legend()
        plt.title(f"Loss curves ({hz})")

        if save_dir:
            plt.savefig(os.path.join(save_dir, f"loss_{hz}.png"))
        plt.close()
