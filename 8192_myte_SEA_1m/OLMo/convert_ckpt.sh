#!/bin/bash

source /home/svu/mrqorib/envs/olmo/bin/activate

cd olmo_data

ckpt_dir=/scratch/Projects/CFP-01/CFP01-CF-060/checkpoints/OLMo-TinyLlama/TinyLlamaV2-1B-stage1

for i in {75..210..5}
do
	origin="${ckpt_dir}/step${i}000-unsharded"
	out_dir="${ckpt_dir}/hf/step${i}000-unsharded"
	echo "Converting $origin to $out_dir ..."
	python ../scripts/convert_olmo2_to_hf.py --input_dir $origin --output_dir $out_dir 2>&1 | tee log_convert.log
done

cd ..
