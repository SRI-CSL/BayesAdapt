g=1
c=2
t=12

#model=meta-llama/Llama-2-7b-chat-hf
#model=Qwen/Qwen2.5-7B
model=Qwen/Qwen2.5-7B


for task in winogrande_s
do
#for gpuid_seed in "0 0" "1 1" "2 2" "3 3"
#for gpuid_seed in "0 4" "1 5" "2 6" "3 7"
for gpuid_seed in "2 6"
do
for laplace_sub in all
do
for laplace_hessian in kron 
do
for laplace_prior in homo 
do
for laplace_optim_step in 100
do
    read gpuid seed <<< "$gpuid_seed"
    accelerate launch --gpu_ids $gpuid run_gpt_laplace.py \
    --model_name_or_path $model \
    --task_name $task \
    --seed $seed \
    --laplace_sub $laplace_sub \
    --laplace_hessian $laplace_hessian \
    --laplace_prior $laplace_prior \
    --laplace_optim_step $laplace_optim_step \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_length 300 \
    --testing_set val \
    --lm_head  &
done
done
done
done
done
wait
done
