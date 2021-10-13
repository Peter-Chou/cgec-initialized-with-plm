#!/bin/bash

scripts_dir=$(dirname $(realpath $0))
project_dir=$(dirname $scripts_dir)

source ${scripts_dir}/utils.sh
test_args $@

train_dir="${project_dir}/experiments/${TEST_MODEL_NAME}"
if [ ! -d "$train_dir" ]; then
    echo "${train_dir} doesn't exist. please check whether model name: ${TEST_MODEL_NAME} is correct."
fi

test_file="${train_dir}/eval_${TEST_MODEL_NAME}.py"
# output_dir="${project_dir}/output/${TEST_MODEL_NAME}/${TEST_MODE}"
# log_file="${output_dir}/test_result_${TEST_MODE}.log"

if [ $TEST_MODE == "preselect" ]; then
    CUDA_VISIBLE_DEVICES=${GPU} python -u ${test_file} \
			--preselect \
			--ckpt=${CKPT}
else
    CUDA_VISIBLE_DEVICES=${GPU} python -u ${test_file} \
			--ckpt=${CKPT}
fi
