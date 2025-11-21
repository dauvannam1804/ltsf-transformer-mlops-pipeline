import matplotlib
matplotlib.use("Agg")  # tránh lỗi khi không có GUI (Airflow container)
import matplotlib.pyplot as plt

def plot_single_loss_curve(train_losses, val_losses, figure_path):
    """
    Plot train/validation loss curves and save to file.

    Args:
        train_losses (list): loss từng epoch của train
        val_losses (list): loss từng epoch của val
        figure_path (str): đường dẫn file PNG để lưu hình
    """
    plt.figure(figsize=(6, 4))

    plt.plot(train_losses, label="Train Loss", antialiased=False)
    plt.plot(val_losses, label="Val Loss", antialiased=False)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Giảm độ nặng ảnh
    plt.savefig(
        figure_path,
        dpi=80,
        bbox_inches="tight",
    )
    plt.close()
