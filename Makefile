# Define variables
PYTHON = python3
PIP = pip
SCRIPT = main.py
TEST_SCRIPT = tests.py

.PHONY: install run tests clean

# Install all dependencies
install:
	$(PIP) install -r requirements.txt

# Run the full training and prediction pipeline
run:
	$(PYTHON) $(SCRIPT)

# Run the test suite to verify the environment and model build
tests:
	$(PYTHON) $(TEST_SCRIPT)

# Remove temporary files and outputs
clean:
	rm -rf output/
	rm -f submission.csv
	find . -type d -name "__pycache__" -exec rm -rf {} +
