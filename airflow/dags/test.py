from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

# ---- Import thử module trong src ----
def test_import():
    print(">>> Bắt đầu import thử module src ...")
    from src.models.linear import Linear
    # from models.linear import Linear
    print(">>> IMPORT OK! Class Linear =", Linear)

with DAG(
    dag_id="test_import_src",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    tags=["debug"],
):
    run_test = PythonOperator(
        task_id="run_test_import",
        python_callable=test_import,
    )

    run_test