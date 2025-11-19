import os
import pickle
import torch
from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from torch.utils.data import DataLoader

from src.models.linear import Linear
from src.models.dlinear import DLinear
from src.models.nlinear import NLinear
from src.data.loader import load_stock_csv
from src.data.preprocess import (
    create_univariate_datasets,
    create_time_based_splits,
    NormalizedDataset,
)

from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model_normalized
from src.utils.plotting import plot_loss_curves

TEMP_DIR = "data/processed"
os.makedirs(TEMP_DIR, exist_ok=True)

RESULTS_DIR = "data/train_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RAW_DATA_PATH = "data/raw/VIC.csv"
SEQ_LENGTHS = [7, 30, 120, 480]
PRED_LEN = 7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =====================================================================
# Task 1 — Load, create datasets, split, normalize, and save to disk
# =====================================================================
def load_and_prepare_data(**context):
    df = load_stock_csv(RAW_DATA_PATH)

    # 1. Create datasets
    datasets = create_univariate_datasets(df, SEQ_LENGTHS, PRED_LEN, "close_log")
    # datasets_path = os.path.join(TEMP_DIR, "datasets.pkl")
    # pickle.dump(datasets, open(datasets_path, "wb"))

    # 2. Create splits
    data_splits = {}
    for seq_name, dataset in datasets.items():
        train, val, test = create_time_based_splits(dataset)
        data_splits[seq_name] = {
            "train": train,
            "val": val,
            "test": test,
        }
    # splits_path = os.path.join(TEMP_DIR, "splits.pkl")
    # pickle.dump(data_splits, open(splits_path, "wb"))

    # 3. Normalize datasets
    normalized_datasets = {}
    scaler = None
    for horizon, dataset in datasets.items():
        nd = NormalizedDataset(dataset, scaler)
        if scaler is None:
            scaler = nd.scaler
        normalized_datasets[horizon] = nd

    # normalized_datasets_path = os.path.join(TEMP_DIR, "norm_datasets.pkl")
    # pickle.dump(normalized_datasets, open(normalized_datasets_path, "wb"))

    # 4. Create normalized splits
    normalized_splits = {}
    for horizon in datasets.keys():
        total_len = len(normalized_datasets[horizon])
        train_len = int(total_len * 0.7)
        val_len = int(total_len * 0.15)

        train_idx = range(0, train_len)
        val_idx = range(train_len, train_len + val_len)
        test_idx = range(train_len + val_len, total_len)

        normalized_splits[horizon] = {
            "train": torch.utils.data.Subset(normalized_datasets[horizon], train_idx),
            "val": torch.utils.data.Subset(normalized_datasets[horizon], val_idx),
            "test": torch.utils.data.Subset(normalized_datasets[horizon], test_idx),
        }

    normalized_splits_path = os.path.join(TEMP_DIR, "norm_splits.pkl")
    pickle.dump(normalized_splits, open(normalized_splits_path, "wb"))

    # Save scaler
    scaler_path = os.path.join(TEMP_DIR, "scaler.pkl")
    pickle.dump(scaler, open(scaler_path, "wb"))

    # XCom: push all paths
    ti = context["ti"]
    ti.xcom_push(key="splits_path", value=normalized_splits_path)
    ti.xcom_push(key="scaler_path", value=scaler_path)


# =====================================================================
# Task 2 — Train model, save weights and loss curves data
# =====================================================================
def train_models_task(**context):
    print("XCom received:", context)
    ti = context["ti"]
    
    splits_path = ti.xcom_pull(task_ids="load_and_prepare_data", key="splits_path")
    scaler_path = ti.xcom_pull(task_ids="load_and_prepare_data", key="scaler_path")

    print("splits_path:", splits_path)
    print("scaler_path:", scaler_path)


    normalized_data_splits = pickle.load(open(splits_path, "rb"))
    scaler = pickle.load(open(scaler_path, "rb"))

    horizon_configs = {
        '7d': {'seq_len': 7, 'pred_len': 7},
        '30d': {'seq_len': 30, 'pred_len': 7},
        '120d': {'seq_len': 120, 'pred_len': 7},
        '480d': {'seq_len': 480, 'pred_len': 7}
    }

    model_configs = {
        'Linear': {},
        'DLinear': {},
        'NLinear': {},
    }

    for horizon, config in horizon_configs.items():
        seq_len = config['seq_len']
        pred_len = config['pred_len']

        # Create model instances
        model_configs['Linear'][horizon] = {
            'model': Linear(seq_len, pred_len),
            'seq_len': seq_len,
            'pred_len': pred_len
        }

        model_configs['DLinear'][horizon] = {
            'model': DLinear(seq_len, pred_len),
            'seq_len': seq_len,
            'pred_len': pred_len
        }

        model_configs['NLinear'][horizon] = {
            'model': NLinear(seq_len, pred_len),
            'seq_len': seq_len,
            'pred_len': pred_len
        }

    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    trained_paths = {}
    results = {"Linear": {}, "DLinear": {}, "NLinear": {}}
    trained_models = {"Linear": {}, "DLinear": {}, "NLinear": {}}
    loss_history = {"Linear": {}, "DLinear": {}, "NLinear": {}}

    for horizon in ["7d", "30d", "120d", "480d"]:
        train_loader = DataLoader(
        normalized_data_splits[horizon]["train"],
        batch_size=batch_size, shuffle=True, drop_last=True
        )
        val_loader = DataLoader(
        normalized_data_splits[horizon]["val"],
        batch_size=batch_size, shuffle=False, drop_last=True
        )
        test_loader = DataLoader(
        normalized_data_splits[horizon]["test"],
        batch_size=batch_size, shuffle=False, drop_last=True
        )

        for model_name in ["Linear", "DLinear", "NLinear"]:
            print("*"*80)
            print(f"Training {model_name} for horizon {horizon}...")
            model = model_configs[model_name][horizon]["model"]
            trained_model, tr_losses, va_losses = train_model(
                model, train_loader, val_loader,
                num_epochs=num_epochs, lr=learning_rate, device=DEVICE
            )
            loss_history[model_name][horizon] = {"train": tr_losses, "val": va_losses}
            trained_models[model_name][horizon] = trained_model

            # (Tuỳ chọn) lưu kết quả test để tổng hợp cuối cùng, không in
            test_results = evaluate_model_normalized(trained_model, test_loader, scaler,
    DEVICE)
            results[model_name][horizon] = test_results


    # Save loss history
    loss_history_path = os.path.join(RESULTS_DIR, "loss_history.pkl")
    pickle.dump(loss_history, open(loss_history_path, "wb"))

    ti.xcom_push(key="weights", value=trained_paths)
    ti.xcom_push(key="loss_history_path", value=loss_history_path)


    plot_dir = os.path.join(RESULTS_DIR, "loss_plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_loss_curves(loss_history, horizons=["7d", "30d", "120d", "480d"], save_dir=plot_dir)


# =====================================================================
# Task 3 — Evaluate & plot loss curves
# =====================================================================
def eval_and_plot(**context):
    ti = context["ti"]

    weights = ti.xcom_pull(key="weights")
    loss_history_path = ti.xcom_pull(key="loss_history_path")
    splits_path = ti.xcom_pull(key="splits_path")

    normalized_splits = pickle.load(open(splits_path, "rb"))
    loss_history = pickle.load(open(loss_history_path, "rb"))

    # Save result folder
    plot_dir = os.path.join(TEMP_DIR, "loss_plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Plot loss curves
    plot_loss_curves(loss_history, horizons=["7d", "30d", "120d", "480d"], save_dir=plot_dir)


# =====================================================================
# Define DAG
# =====================================================================
with DAG(
    dag_id="ltsf_baseline_pipeline",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
):

    t1 = PythonOperator(
        task_id="load_and_prepare_data",
        python_callable=load_and_prepare_data,
    )

    t2 = PythonOperator(
        task_id="train_models",
        python_callable=train_models_task,
    )

    # t3 = PythonOperator(
    #     task_id="evaluate_and_plot",
    #     python_callable=eval_and_plot,
    # )

    t1 >> t2
