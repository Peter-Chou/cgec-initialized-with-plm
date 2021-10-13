#!/bin/bash

scripts_dir=$(dirname $(realpath $0))
project_dir=$(dirname $scripts_dir)

source ${scripts_dir}/utils.sh
train_args $@

train_dir="${project_dir}/experiments/${MODEL_NAME}"
if [ ! -d "$train_dir" ]; then
    echo "${train_dir} doesn't exist. please check whether model name: ${MODEL_NAME} is correct."
fi

train_file="${train_dir}/train_${MODEL_NAME}.py"
output_dir="${project_dir}/output/${MODEL_NAME}/${MODE}"
mkdir -p $output_dir

echo "start training ${MODEL_NAME} in ${MODE} stage on gpu ${GPU}"

output_file="$output_dir/${MODE}.log"

CUDA_VISIBLE_DEVICES=${GPU} TRANSFORMERS_VERBOSITY=info nohup \
		    python -u ${train_file} \
		    --trainset_ratio=${TRAIN_RATIO}  > ${output_file} 2>&1 &

echo "please see training details in ${output_file}"
