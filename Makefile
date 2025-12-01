clean:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache .dist
	find . -type d -name '.ipynb_checkpoints' -exec rm -rf {} +
	find . -type f -name '.DS_Store' -exec rm -rf {} +

run:
	bash run.sh