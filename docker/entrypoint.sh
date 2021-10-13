#!/bin/bash

TEST_FILE_PATH=$1

TEST_FILE_PATH="/${TEST_FILE_PATH}"
TEST_FILE_NAME=$(echo "${TEST_FILE_PATH}" | cut -f 1 -d '.')
SEGMENTED_FILE_PATH="${TEST_FILE_NAME}_segmented.txt"

echo "start segment file: ${TEST_FILE_PATH}"| tee "${TEST_FILE_NAME}_score_result"
python3 /opt/segment_test_file.py -i ${TEST_FILE_PATH}

echo "start evaluate segmented test file with gold standard annotation" | tee -a "${TEST_FILE_NAME}_score_result"
# gold.01 combines annotator 0(gold.0) and annotator 1(gold.1)
python2 /opt/m2scorer/scripts/m2scorer.py \
	${SEGMENTED_FILE_PATH} \
	/opt/gold/gold.01 | tee -a "${TEST_FILE_NAME}_score_result"
