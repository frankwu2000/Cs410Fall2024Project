.PHONY: build import-requirements export-requirements

build:
	python main.py W1-W6	

clean:
	rm W1-W6_subtitle_data_sbert_segment_output

import-requirements:
	pip install -r requirements.txt

export-requirements:
	pip freeze > requirements.txt