import os
import pickle
import torch
import numpy as np
from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator

from torch.utils.data import DataLoader

from src.models.linear import Linear
from src.models.dlinear import DLinear
from src.models.nlinear import NLinear
from src.models.hybrid_transformer import HybridDLinearTransformer
from src.data.loader import load_stock_csv
from src.data.preprocess import (
    create_univariate_datasets,
    create_time_based_splits,
    NormalizedDataset,
)

from src.training.trainer import train_model
from src.evaluation.evaluator import evaluate_model_normalized
from src.utils.plotting import plot_single_loss_curve

import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

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

    model_configs = { 'Linear': {}, 'DLinear': {}, 'NLinear': {}, 'HLinear': {} }

    # Prepare model configurations
    for horizon, cfg in horizon_configs.items():
        seq_len = cfg['seq_len']
        pred_len = cfg['pred_len']

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
        model_configs['HLinear'][horizon] = {
            'model': HybridDLinearTransformer(seq_len, pred_len),
            'seq_len': seq_len,
            'pred_len': pred_len
        }

    batch_size = 32
    num_epochs = 50
    learning_rate = 0.001

    loss_history = {"Linear": {}, "DLinear": {}, "NLinear": {}, "HLinear": {}}

    mlflow.set_tracking_uri("http://172.17.0.1:5000")
    mlflow.set_experiment("ltsf_baseline")
   
    # Directory for plots
    plot_dir = os.path.join(RESULTS_DIR, "loss_plots")
    os.makedirs(plot_dir, exist_ok=True)

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

        for model_name in ["Linear", "DLinear", "NLinear", "HLinear"]:
            print("*" * 80)
            print(f"Training {model_name} for horizon {horizon}...")
            model_cfg = model_configs[model_name][horizon]

            model = model_cfg["model"]
            seq_len = model_cfg["seq_len"]
            pred_len = model_cfg["pred_len"]

            with mlflow.start_run(run_name=f"{model_name}_{horizon}"):

                # Log hyperparameters
                mlflow.log_params({
                    "model": model_name,
                    "horizon": horizon,
                    "seq_len": seq_len,
                    "pred_len": pred_len,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "epochs": num_epochs,
                })

                # Log model config
                mlflow.log_dict(model_cfg, "model_config.json")

                # Train model
                trained_model, tr_losses, va_losses = train_model(
                    model, train_loader, val_loader,
                    num_epochs=num_epochs, lr=learning_rate, device=DEVICE
                )

                # Save losses locally
                loss_history[model_name][horizon] = {
                    "train": tr_losses,
                    "val": va_losses
                }

                # Log final losses
                mlflow.log_metric("train_loss_last", float(tr_losses[-1]))
                mlflow.log_metric("val_loss_last", float(va_losses[-1]))
                mlflow.log_metric("val_loss_best", float(min(va_losses)))

                # EVALUATE
                test_results = evaluate_model_normalized(
                    trained_model, test_loader, scaler, DEVICE
                )

                scalar_metrics = {k: v for k, v in test_results.items() if np.isscalar(v)}
                mlflow.log_metrics({f"test_{k}": float(v) for k, v in scalar_metrics.items()})

                # Save loss history for this model
                mlflow.log_dict(
                    {"train": list(map(float, tr_losses)),
                     "val": list(map(float, va_losses))},
                    "loss_history.json"
                )

                # Plot loss curves
                fig_path = os.path.join(plot_dir, f"{model_name}_{horizon}.png")
                plot_single_loss_curve(tr_losses, va_losses, figure_path=fig_path)
                mlflow.log_artifact(fig_path)

                # Log scaler artifact
                mlflow.log_artifact(scaler_path)

                # Prepare 1 sample input for signature
                batch = next(iter(train_loader))[0][:1]     # tensor [1, seq_len, features]
                batch = batch.to(torch.float32)

                input_example = batch.cpu().numpy()

                # Create signature for MLflow
                signature = infer_signature(
                    input_example,
                    trained_model(batch.to(DEVICE)).detach().cpu().numpy()
                )

                # Log model with signature + correct dtype input example
                mlflow.pytorch.log_model(
                    trained_model,
                    name=f"{model_name}_{horizon}",
                    signature=signature,
                    input_example=input_example.astype(np.float32)
                )

    # Save full loss history
    loss_history_path = os.path.join(RESULTS_DIR, "loss_history.pkl")
    pickle.dump(loss_history, open(loss_history_path, "wb"))

    ti.xcom_push(key="loss_history_path", value=loss_history_path)




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

    t1 >> t2
