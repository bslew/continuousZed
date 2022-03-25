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

dw_tstruct_data:
	mysql -uroot -p -h galaxy.astro.int kra -e 'select * from struct_temp_tewa_avg where dt>"2022-01-01 00:00:00"' | tr '\t' ',' > data/Tstruct.csv
	gzip data/Tstruct.csv

cat_data:
	more data/FastScan_*.off

send_zed_to_rt4:
	source ${VENV}/bin/activate && continuousZed.py --set_dZD_auto
	source ${VENV}/bin/activate && continuousZed.py --set_dxZD_auto

zed: dw_crossscan_data send_zed_to_rt4

calc_median:
	source ${VENV}/bin/activate && continuousZed.py --median ${VERB}

export_data:
	@echo "export unified corrections for pointing model fitting"
	@echo "the pointing corrections will be reduced to corrections"
	@echo "as would be measured using the ....what..?"
	source ${VENV}/bin/activate && continuousZed.py --median --export_suff _since20220211 --start_time "2022-02-11 00:00:00"  ${VERB}
#	mv data/corrections_stats.unified.pkl data/corrections_since20220211_stats.unified.pkl
#	rename corrections_C.unified.jpg
		
sync_web_plots: dw_crossscan_data export_data calc_median 
	-cp -p data/*.jpg data/*.pkl /home/rt32time/rt32time/data
	-cp -p data/corrections_* /home/rt32time/rt32time/data
	-cp -p data/corrections-* /home/rt32time/rt32time/data

copy_web_plots_devel: 
	-cp -p data/*.jpg data/*.pkl ~/programy/rt4/time-distro/time-client-app/client/rt32time/data
	-cp -p data/corrections_* ~/programy/rt4/time-distro/time-client-app/client/rt32time/data
	-cp -p data/corrections-* ~/programy/rt4/time-distro/time-client-app/client/rt32time/data
	