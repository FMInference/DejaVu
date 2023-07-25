task_type=$1
model_name=$2

cd ..

python global_user_client.py --op estimate --task-type $task_type --model-name $model_name