.PHONY: install install-dev clean build test lint docs docker

install:
	pip install .

install-dev:
	pip install -e .[dev,docs]

install-gpu:
	pip install -e .[gpu,dev,docs]

install-cpu:
	pip install -e .[cpu,dev,docs]

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".tox" -exec rm -rf {} +
	find . -type d -name ".eggs" -exec rm -rf {} +

build: clean
	python setup.py sdist bdist_wheel

test:
	pytest tests/ --cov=uhg --cov-report=term-missing

lint:
	black .
	isort .
	flake8 .
	mypy uhg

docs:
	cd docs && make html

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

publish: clean
	python setup.py sdist bdist_wheel
	twine check dist/*
	twine upload dist/*

publish-test: clean
	python setup.py sdist bdist_wheel
	twine check dist/*
	twine upload --repository-url https://test.pypi.org/legacy/ dist/* 