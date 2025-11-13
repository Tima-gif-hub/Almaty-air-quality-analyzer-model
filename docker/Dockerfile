FROM python:3.11-slim

ENV TZ=Asia/Almaty \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y build-essential git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/
COPY Makefile .env.example README.md ./

EXPOSE 8501

ENV PYTHONPATH=/app

CMD ["bash","-lc","python -m src.wait_for_db && streamlit run src/app_streamlit.py --server.port 8501 --server.address 0.0.0.0"]
