ENV_FILE ?= .env
PYTHON ?= python
DOCKER_COMPOSE ?= docker compose -f docker/docker-compose.yml
HORIZON ?= $(FORECAST_HORIZON)

ifneq (,$(wildcard $(ENV_FILE)))
include $(ENV_FILE)
export
endif

.PHONY: install data clean features train eval forecast explain app docker-build docker-up docker-down

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.cli ingest --path $(DATA_PATH)

clean:
	$(PYTHON) -m src.cli clean

features:
	$(PYTHON) -m src.cli make-features

train:
	$(PYTHON) -m src.cli train --model rf

eval:
	$(PYTHON) -m src.cli eval

forecast:
	$(PYTHON) -m src.cli forecast --model rf --h $(HORIZON)

explain:
	$(PYTHON) -m src.cli explain --model rf

app:
	streamlit run src/app_streamlit.py --server.port 8501 --server.address 0.0.0.0

docker-build:
	$(DOCKER_COMPOSE) build

docker-up:
	$(DOCKER_COMPOSE) up

docker-down:
	$(DOCKER_COMPOSE) down
