g=1
c=1
t=6

#/user/work/ad20999/infrastructure/blue_pebble/bin/lbatch -m 16 -c $c -g $g -t $t --cmd accelerate launch run_gpt.py \
#model=meta-llama/Llama-2-7b-chat-hf
model=Qwen/Qwen2.5-7B

#for task in ARC-Challenge
#for task in winogrande_s ARC-Easy winogrande_m openbookqa boolq  
#for task in ARC-Challenge openbookqa boolq
for task in winogrande_s
do
#for gpuid_seed in "0 0" "1 1" "2 2" "3 3"
#for gpuid_seed in "0 4" "1 5" "2 6" "3 7"
for gpuid_seed in "2 6"
do
    read gpuid seed <<< "$gpuid_seed"
    accelerate launch --gpu_ids $gpuid run_gpt.py \
    --model_name_or_path $model \
    --task_name $task \
    --seed $seed \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_length 300 \
    --testing_set val \
    --lm_head &
done
wait
done
