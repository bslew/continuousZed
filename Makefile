PROG=continuousZed
SHELL=/bin/bash
VERB=
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
	install sh/getRT32_ROH_data ${HOME}/bin/

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

dw_data: dw_crossscan_data dw_rt4_data

dw_crossscan_data:
	-mkdir data data.bak
	cp data/*.off data/*.txt data.bak
	-scp oper@annaring:/home/oper/scan/FastScan_5.00.off data
	-scp oper@annaring:/home/oper/scan/FastScan_6.70.off data
	-scp oper@annaring:/home/oper/scan/FastScan_12.00.off data
	-scp oper@annaring:/home/oper/scan/FastScan_22.00.off data
	-cat data/FastScan_5.00.off | sed -e '/^#/d' | wc -l
	-cat data/FastScan_6.70.off | sed -e '/^#/d' | wc -l
	-cat data/FastScan_12.00.off | sed -e '/^#/d' | wc -l
	-cat data/FastScan_22.00.off | sed -e '/^#/d' | wc -l

dw_rt4_data:
	-scp rt32time@galaxy:~/continuousZed/data/continuous_corrections.txt data
	-scp rt32time@galaxy:~/continuousZed/data/ROH.? data

cat_data:
	more data/FastScan_*.off

send_zed_to_rt4:
	source ${VENV}/bin/activate && continuousZed.py --set_dZD_auto
	source ${VENV}/bin/activate && continuousZed.py --set_dxZD_auto

zed: dw_crossscan_data send_zed_to_rt4

calc_median:
	source ${VENV}/bin/activate && continuousZed.py --median ${VERB}
	
sync_web_plots: dw_crossscan_data calc_median
	-cp -p data/*.jpg data/*.pkl /home/rt32time/rt32time/data

