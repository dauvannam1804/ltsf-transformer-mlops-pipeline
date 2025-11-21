# TODO List -- Spec-Driven Development cho Dự án LTSF + Airflow + MLflow

## 1. Mục tiêu

-   Xây dựng pipeline LTSF (Linear, DLinear, NLinear → cải tiến với
    Transformer head).
-   Thêm nhánh Transformer sau 2 nhánh trend/seasonal của DLinear.
-   Tự động hóa toàn bộ vòng đời training bằng **Airflow**.
-   Theo dõi thí nghiệm và so sánh mô hình bằng **MLflow**.
-   Viết blog đi kèm để người đọc:
    -   Hiểu thiết kế spec-driven.
    -   Chạy được code.
    -   So sánh baseline vs cải tiến.

------------------------------------------------------------------------

## 2. Spec tổng thể của dự án

### 2.1. Thành phần chính

1.  **Module Data**
    -   Chuẩn bị dữ liệu giá cổ phiếu (hoặc dataset tùy chọn).\
    -   Tách train/val/test theo rolling window.\
    -   Chuẩn hóa dữ liệu theo từng mô hình
        (Linear/NLinear/DLinear/Hybrid).
2.  **Module Models**
    -   Mô hình Linear.
    -   Mô hình NLinear.
    -   Mô hình DLinear.
    -   **Mô hình Hybrid: DLinear + Transformer Head**
        -   Trend → Linear Head\
        -   Seasonal → Linear Head\
        -   Tổng hợp → Transformer → Output Layer
3.  **Module Training**
    -   Train 4 mô hình.
    -   Lưu lại weights + metrics.
    -   MLflow autolog: loss, MAE, RMSE, charts.
4.  **Module Evaluation**
    -   So sánh mô hình bằng MAE, RMSE, MAPE.
    -   Lưu biểu đồ dự đoán vào MLflow.
5.  **Pipeline Airflow**
    -   Task 1: Fetch data
    -   Task 2: Preprocess
    -   Task 3: Train baseline (Linear, NLinear, DLinear)
    -   Task 4: Train Hybrid Transformer
    -   Task 5: Evaluate & compare
    -   Task 6: Log to MLflow
    -   Task 7: Upload Artifacts (plots, weights)
    -   Trigger: daily hoặc manual
6.  **Blog Structure**
    -   Giải thích LTSF.
    -   Tại sao Linear + DLinear hiệu quả.
    -   Nhược điểm → động lực thêm Transformer.
    -   Thiết kế hybrid.
    -   Tại sao Airflow phù hợp.
    -   Tại sao MLflow giúp minh bạch so sánh.
    -   Kết quả và biểu đồ.

------------------------------------------------------------------------

## 3. Danh sách công việc chi tiết (Spec-driven TODO)

### 3.1. Requirements & Setup

-   [X] Tạo môi trường Conda / Virtualenv.
-   [X] Cài đặt thư viện: PyTorch, MLflow, pandas, Airflow, matplotlib.
-   [X] Tạo repository theo cấu trúc:

```{=html}
<!-- -->
```
    project/
      data/
      airflow/
      mlflow/
      src/
        data/
        models/
        training/
        evaluation/
      blog/

------------------------------------------------------------------------

### 3.2. Data Module

-   [X] Viết script `load_data.py`.
-   [X] Viết script `preprocess.py`.
-   [X] Triển khai rolling window tạo dataset.
-   [ ] Lưu version dataset bằng MLflow (optional).

------------------------------------------------------------------------

### 3.3. Baseline Model Module

-   [X] Viết lớp `LinearModel`.
-   [X] Viết lớp `NLinearModel`.
-   [X] Viết lớp `DLinearModel`.
-   [X] Viết hàm train/eval chung.

------------------------------------------------------------------------

### 3.4. Transformer Hybrid Module

-   [X] Thiết kế kiến trúc:
    -   Input = concat(trend_out, seasonal_out)
    -   Transformer Encoder (1--2 layers)
    -   Output Layer → forecast
-   [X] Viết lớp `HybridDLinearTransformer`.
-   [X] Viết script train hybrid model.

------------------------------------------------------------------------

### 3.5. Training Module + MLflow

-   [x] Viết `train.py`:
    -   [x] Baseline training loop
    -   [x] Hybrid training loop
    -   [ ] MLflow experiment tracking
-   [x] Log metrics:
    -   Train/Val loss
    -   MAE, RMSE, MAPE
-   [ ] Log artifacts:
    -   Forecast charts
    -   Model weights
    -   Loss curves

------------------------------------------------------------------------

### 3.6. Evaluation Module

-   [ ] Viết script `evaluate.py`.
-   [ ] Tạo bảng so sánh 4 mô hình.
-   [ ] Tạo biểu đồ:
    -   Actual vs Predict
    -   Error distribution
-   [ ] Log vào MLflow.

------------------------------------------------------------------------

### 3.7. Airflow Pipeline

#### DAG: `ltsf_training_pipeline`

-   [ ] Task 1: `fetch_data`
-   [ ] Task 2: `preprocess_data`
-   [ ] Task 3: `train_baseline_models`
-   [ ] Task 4: `train_hybrid_model`
-   [ ] Task 5: `evaluate_models`
-   [ ] Task 6: `log_results_mlflow`
-   [ ] Task 7: `upload_artifacts`
-   [ ] Validate dependencies:\
    fetch → preprocess → train baseline → train hybrid → evaluate → log

#### Lý do dùng Airflow:

-   Tự động hóa training theo lịch.
-   Dễ thêm nhiệm vụ (data drift / retrain).
-   Dễ debug (UI trực quan).
-   Dễ mở rộng sang multistock / multipipeline.

------------------------------------------------------------------------

## 4. Cách bạn có thể viết blog sau khi hoàn thành

-   [ ] Xuất toàn bộ notebooks thành hình minh họa.
-   [ ] Dùng file MLflow để embed kết quả.
-   [ ] Mô tả rõ ràng:
    -   Baseline Linear vs DLinear khác gì?
    -   Transformer cải thiện ở điểm nào?
    -   Airflow orchestration flow (kèm hình DAG).
    -   Vì sao MLflow phù hợp cho dự án LTSF?

------------------------------------------------------------------------

## 5. Checklist hoàn thiện dự án

-   [ ] Repo sạch và tái lập được.
-   [ ] Airflow DAG chạy OK.
-   [ ] MLflow logs đầy đủ.
-   [ ] Có notebook demo.
-   [ ] Blog có thể xuất bản.

------------------------------------------------------------------------

## 6. Kết luận

Tài liệu này đóng vai trò **bản đặc tả tổng thể (spec-driven)** để triển
khai dự án LTSF nâng cao chuẩn MLOps, giúp người đọc: - Hiểu dự án từ
kiến trúc → pipeline → training → deployment. - Dễ dàng tái tạo kết
quả. - Tập trung vào cải tiến mô hình thay vì lỗi quy trình.
