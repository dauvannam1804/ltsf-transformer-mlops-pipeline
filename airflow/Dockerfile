FROM apache/airflow:3.1.3

# Airflow yêu cầu chạy pip dưới user "airflow"
USER airflow

# Copy requirements (bao gồm matplotlib và các package khác)
COPY requirements.txt .

# Cài packages từ requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Cài PyTorch CPU
RUN pip install --no-cache-dir torch \
    --index-url https://download.pytorch.org/whl/cpu
