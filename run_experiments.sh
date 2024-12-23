dataset_name=all
temperature=0.6
batch_size=16
max_new_tokens=512
proxy_base_url=http://127.0.0.1:4000
output_folder=./medhalt/predictions/
timestamp=$(date +%s)
experiment_name="testing_litellm"

declare -a models=(
	# "gemini-1.5-pro"
	"gpt-3.5-turbo"
)

for model in "${models[@]}"
do
	echo "Running prediction for ${model}"
	python3 -m medhalt.models.model --model_name=$model --model_path=./$experiment_name/$timestamp \
					--dataset_name=$dataset_name \
					--temperature=$temperature \
					--batch_size=$batch_size \
					--max_new_token=$max_new_tokens \
					--proxy_base_url=$proxy_base_url \
					--output_folder=$output_folder
done
