
.PHONY: page clean_page extract genQA checkQA clean_data clean_all

PYTHON = /home/liusiyuan/.conda/envs/Ti/bin/python

SCRIPT_EXTRACT = scripts/extract_facts.py
SCRIPT_GENQA = scripts/genQA.py
SCRIPT_CHECKQA = scripts/checkQA.py

ACTION = upload
# ACTION = download

page:
	$(PYTHON) scripts/page.py

clean_page:
	rm -rf ./pages/*

extract:
	$(PYTHON) $(SCRIPT_EXTRACT) --action $(ACTION)

genQA:
	$(PYTHON) $(SCRIPT_GENQA) --action $(ACTION)

checkQA:
	$(PYTHON) $(SCRIPT_CHECKQA) --action $(ACTION)

clean_data:
	rm -rf ./batch/*
	rm -rf ./dataset/*

clean_all: clean_page clean_data





