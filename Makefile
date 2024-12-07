.PHONY: build import-requirements export-requirements

build:
	python main.py

import-requirements:
	pip install -r requirements.txt

export-requirements:
	pip freeze > requirements.txt