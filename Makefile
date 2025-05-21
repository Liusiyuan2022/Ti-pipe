
.PHONY: page clean_page clean_data clean_all

PYTHON = /home/liusiyuan/.conda/envs/Ti/bin/python

page:
	$(PYTHON) scripts/page.py
clean_page:
	rm -rf ./pages/*

clean_data:
	rm -rf ./dataset/*

clean_all: clean_page clean_data





