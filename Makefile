.PHONY: init setup train test lint clean

# Variables
PYTHON = uv run
SRC_DIR = src
TEST_DIR = tests
PAPER_DIR = papers

# Usage: make init PAPER=seq2seq
init:
ifndef PAPER
	$(error PAPER variable is not set. Usage: make init PAPER=your_paper_name)
endif
	@echo "Initializing structure for paper: $(PAPER)..."
	# Create Paper Implementation directory
	mkdir -p $(PAPER_DIR)/$(PAPER)
	
	# Create Core Source directories (if not exists)
	mkdir -p $(SRC_DIR)/layers $(SRC_DIR)/training $(SRC_DIR)/utils
	mkdir -p $(TEST_DIR)
	
	# Initialize __init__.py files
	@touch $(SRC_DIR)/layers/__init__.py $(SRC_DIR)/training/__init__.py $(SRC_DIR)/utils/__init__.py
	@touch $(PAPER_DIR)/$(PAPER)/__init__.py
	
	# Create essential files for the paper module
	@touch $(PAPER_DIR)/$(PAPER)/model.py \
	       $(PAPER_DIR)/$(PAPER)/trainer.py \
	       $(PAPER_DIR)/$(PAPER)/inference.py \
	       $(PAPER_DIR)/$(PAPER)/config.py
	
	@echo "Structure for $(PAPER) initialized successfully."

# Setup Environment
setup:
	@echo "Syncing dependencies..."
	uv sync

# Run Training (Generic: targets specific paper)
train:
ifndef PAPER
	$(error PAPER variable is not set. Usage: make train PAPER=seq2seq)
endif
	@echo "Executing training pipeline for $(PAPER)..."
	$(PYTHON) train.py --paper $(PAPER)

# Run Tests
test:
	@echo "Running unit tests and gradient checks..."
	$(PYTHON) -m pytest $(TEST_DIR)

# Linting
lint:
	@echo "Linting and formatting..."
	uv run ruff check .
	uv run ruff format --check .

# Clean
clean:
	@echo "Cleaning up..."
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +