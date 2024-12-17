#!/bin/bash
datasets_folder=$1
prediction_folder=$2

declare -a folders=(
				"gemini",
				)

for key in "${folders[@]}":
do
	echo "Running prediction for ${key}"
	python3 evaluate.py \
		--prediction_folder=${prediction_folder}/${key} \
		--dataset_folder=${datasets_folder} \
		--do_json_conversion
		--rest_client http://localhost:8000/v1
done
