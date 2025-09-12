# Khet Guard ML Training Makefile

.PHONY: help install data-prep train-plant train-cattle train-hpo evaluate export clean

# Default target
help:
	@echo "Available targets:"
	@echo "  install       - Install dependencies"
	@echo "  data-prep     - Prepare datasets"
	@echo "  train-plant   - Train plant disease model"
	@echo "  train-cattle  - Train cattle breed model"
	@echo "  train-hpo     - Run hyperparameter optimization"
	@echo "  evaluate      - Evaluate trained model"
	@echo "  export        - Export model to production formats"
	@echo "  clean         - Clean output directories"

# Variables
DATA_DIR := ../../
OUTPUT_DIR := ./outputs
PROCESSED_DATA_DIR := ./processed_data
CONFIG_DIR := ./configs

# Install dependencies
install:
	pip install -r requirements.txt

# Prepare datasets
data-prep:
	python data_prep.py --data_dir $(DATA_DIR) --output_dir $(PROCESSED_DATA_DIR)

# Train plant disease model
train-plant:
	python train.py --config $(CONFIG_DIR)/plant_disease_config.yaml --data_dir $(PROCESSED_DATA_DIR) --output_dir $(OUTPUT_DIR)/plant_disease

# Train cattle breed model
train-cattle:
	python train.py --config $(CONFIG_DIR)/cattle_breed_config.yaml --data_dir $(PROCESSED_DATA_DIR) --output_dir $(OUTPUT_DIR)/cattle_breed

# Run hyperparameter optimization
train-hpo:
	python train.py --config $(CONFIG_DIR)/plant_disease_config.yaml --data_dir $(PROCESSED_DATA_DIR) --output_dir $(OUTPUT_DIR)/plant_disease_hpo --hpo

# Evaluate model
evaluate:
	python evaluate.py --checkpoint $(OUTPUT_DIR)/plant_disease/best.ckpt --config $(OUTPUT_DIR)/plant_disease/model_info.json --data_dir $(PROCESSED_DATA_DIR)

# Export model
export:
	python export.py --checkpoint $(OUTPUT_DIR)/plant_disease/best.ckpt --config $(OUTPUT_DIR)/plant_disease/model_info.json --data_dir $(PROCESSED_DATA_DIR) --output_dir $(OUTPUT_DIR)/exported_models --benchmark

# Clean output directories
clean:
	rm -rf $(OUTPUT_DIR)
	rm -rf $(PROCESSED_DATA_DIR)
	rm -rf evaluation_plots
	rm -rf logs

# Docker targets
docker-build:
	docker build -t khet-guard-training .

docker-run:
	docker run --gpus all -v $(PWD):/app -w /app khet-guard-training make train-plant

# Full pipeline
pipeline: install data-prep train-plant evaluate export
	@echo "Full training pipeline completed!"

# Quick test with small dataset
test-pipeline: install data-prep
	python train.py --config $(CONFIG_DIR)/plant_disease_config.yaml --data_dir $(PROCESSED_DATA_DIR) --output_dir $(OUTPUT_DIR)/test --max_epochs 2
