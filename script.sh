#!/bin/bash

lr=$1
sample_n_grad=$2
sample_n_var=$3

python main_run.py res/ bm-sns.py   10000 $lr $sample_n_grad $sample_n_var  1000 100 run,plot
python main_run.py res/ bm-tcl.py   10000 $lr $sample_n_grad $sample_n_var  1000 100 run,plot
python main_run.py res/ bm-time.py  10000 $lr $sample_n_grad $sample_n_var  1000 100 run,plot
