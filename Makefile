.PHONY: help install train evaluate copy-logs api docker-build docker-run clean

help:
	@echo "LLM Framework Operations"
	@echo "------------------------"
	@echo "install      - Install Python dependencies locally"
	@echo "train        - Run the default customer support training experiment via CLI"
	@echo "evaluate     - Evaluate the default experiment via CLI"
	@echo "insights     - Generate automated decision intelligence insights via CLI"
	@echo "compare      - Run the comparator over two sample runs"
	@echo "api          - Start the FastAPI inference service locally"
	@echo "docker-build - Build the LLM inference Docker image"
	@echo "docker-run   - Serve the API using docker-compose"
	@echo "clean        - Remove python cache and temporary files"

install:
	pip install -r requirements.txt

train:
	python -m src.cli train --config configs/exp001_customer_support.yaml

evaluate:
	python -m src.cli evaluate --run-id exp001_customer_support

insights:
	python -m src.cli insights

compare:
	python -m src.cli compare --runs exp001_customer_support exp001_customer_support

api:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-run:
	docker-compose -f docker/docker-compose.yml up -d

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
