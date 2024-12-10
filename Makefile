.PHONY: build import-requirements export-requirements

build:
	python main.py W1-W6	

clean:
	rm W1-W6_subtitle_data_sbert_segment_output

clean-build:
	make clean
	make build	

import-requirements:
	pip install -r requirements.txt

export-requirements:
	pip freeze > requirements.txt