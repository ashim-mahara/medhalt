model=$1
dataset_name=all
temperature=${2-0.6}
batch_size=${3-4}
max_new_tokens=${4-128}
rest_client=${5-http://127.0.0.1:8080/generate}
output_folder=${6-./medhalt/predictions/}
timestamp=$(date +%s)

python3 -m medhalt.models.model --model_path=./gemini-$timestamp \
                 --dataset_name=$dataset_name \
                 --temperature=$temperature \
                 --batch_size=128 \
                 --max_new_token=$max_new_tokens \
                 --rest_client=$rest_client \
                 --output_folder=$output_folder
