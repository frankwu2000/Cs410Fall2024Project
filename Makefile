.PHONY: build import-requirements export-requirements

build:
	python main.py

clean:
	rm W1_subtitle_data_sbert_segment_output

import-requirements:
	pip install -r requirements.txt

export-requirements:
	pip freeze > requirements.txt