PROG=continuousZed
SHELL=/bin/bash

VENV=venv

#check_dir:
#	-mkdir -p ${LOGDIR}

install: 
	@echo "Installing python package"
	source ${VENV}/bin/activate && cd python && python setup.py install
#	source ${VENV}/bin/activate && cp sh/share_wisdom.sh ${VENV}/bin

	@echo "Installing scripts"
#	install sh/continuousZed venv/bin/
	install sh/continuousZed ${HOME}/bin/

	@echo "Installing config file"
	-mkdir -p $(HOME)/.config/${PROG}/
	@echo "This will override your current config file ($(HOME)/.config/${PROG}/${PROG}.ini)."
	@echo "The original file will be moved to $(HOME)/.config/${PROG}/${PROG}.ini.bak"
	@echo "Press enter to continue..."
	@read
	@if [ -f $(HOME)/.config/${PROG}/${PROG}.ini ]; then mv $(HOME)/.config/${PROG}/${PROG}.ini $(HOME)/.config/${PROG}/${PROG}.ini.bak ; fi
	cp ${PROG}.ini $(HOME)/.config/${PROG}/${PROG}.ini
	@echo "done"


help:
	@echo ""
	@echo "USAGE"
	@echo ""
	@echo ""
	@echo ""
	@echo ""

dw_data:
	-mkdir data
	-scp oper@annaring:/home/oper/scan/FastScan_5.00.off data
	-scp oper@annaring:/home/oper/scan/FastScan_6.70.off data
	-scp oper@annaring:/home/oper/scan/FastScan_12.00.off data
	-scp oper@annaring:/home/oper/scan/FastScan_22.00.off data

	-scp rt32time@galaxy:~/continuousZed/data/continuous_corrections.txt data

cat_data:
	more data/FastScan_*.off

send_zed_to_rt4:
	source ${VENV}/bin/activate && continuousZed.py --set_dZD_auto

zed: dw_data send_zed_to_rt4

calc_median:
	source ${VENV}/bin/activate && continuousZed.py --median

