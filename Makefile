PYTHON ?= python
QUICK_DATASETS = f1_l-d_kp_10_269 f2_l-d_kp_20_878 f5_l-d_kp_15_375 f8_l-d_kp_23_10000 \
	knapPI_1_100_1000_1 knapPI_1_200_1000_1

.PHONY: install run run-quick test

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) main.py experiments

run-quick:
	$(PYTHON) main.py experiments --datasets $(QUICK_DATASETS)

test:
	$(PYTHON) -m unittest discover -s tests -v
