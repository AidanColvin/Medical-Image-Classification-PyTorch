export PYTHONPATH := $(shell pwd)

.PHONY: test run submit full-pipeline clean

test:
	pytest tests/test_suite.py -W ignore::RuntimeWarning

run:
	python3 main.py

submit:
	python3 scripts/generate_submission.py

full-pipeline: test
	@echo "Tests Passed. Starting Training..."
	python3 main.py
	@echo "Training Complete. Generating Submission..."
	python3 scripts/generate_submission.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
