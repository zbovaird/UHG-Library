.PHONY: install install-dev clean build test lint type docs docker

install:
	pip install .

install-dev:
	pip install -e ".[dev]"

install-docs:
	pip install -e ".[docs]"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".tox" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".eggs" -exec rm -rf {} + 2>/dev/null || true

build: clean
	python -m build

test:
	pytest --cov=uhg --cov-report=term-missing

lint:
	black uhg tests

type:
	mypy uhg

docs:
	mkdocs build -f mkdocs.yml

docs-serve:
	mkdocs serve -f mkdocs.yml

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

publish: clean build
	twine check dist/*
	twine upload dist/*

publish-test: clean build
	twine check dist/*
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*
