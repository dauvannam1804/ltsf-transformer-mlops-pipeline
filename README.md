# LTSF Transformer MLOps Pipeline

A spec-driven project for Long-Term Time Series Forecasting (LTSF) that extends the DLinear architecture with a Transformer head. This project includes a complete MLOps pipeline using Airflow.

## Prerequisites

Before you begin, ensure you have the following installed:
- [Docker](https://www.docker.com/) & Docker Compose
- [uv](https://github.com/astral-sh/uv) (for Python package management)
- Python 3.13+

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/dauvannam1804/ltsf-transformer-mlops-pipeline
    cd ltsf-transformer-mlops-pipeline
    ```

2.  **Initialize the environment:**
    This project uses `uv` for dependency management.
    ```bash
    uv sync
    ```
    Or manually:
    ```bash
    uv init
    uv pip install -r requirements.txt # If you have a requirements.txt generated
    # Or rely on pyproject.toml
    ```
    *Note: The project contains a `uv.lock` file, so `uv sync` is the recommended way to install dependencies.*

## Running the Pipeline (Airflow)

The MLOps pipeline is orchestrated using Apache Airflow, running in Docker containers.

1.  **Navigate to the airflow directory:**
    ```bash
    cd airflow
    ```

2.  **Build the Docker images:**
    ```bash
    docker compose build --no-cache
    ```

3.  **Start the services:**
    ```bash
    docker compose up -d
    ```

4.  **Access Airflow:**
    Open your browser and go to `http://localhost:8080`.
    - **Username:** `airflow`
    - **Password:** `airflow` (default)

## Project Structure

- `src/`: Source code for the model and pipeline components.
- `airflow/`: Airflow configuration, DAGs, and Docker setup.
    - `dags/`: Airflow DAG definitions.
    - `docker-compose.yaml`: Docker Compose configuration for Airflow services.
    - `Dockerfile`: Custom Airflow image definition.
- `data/`: Directory for datasets.
- `notebooks/`: Jupyter notebooks for experimentation.
- `pyproject.toml`: Python project configuration and dependencies.