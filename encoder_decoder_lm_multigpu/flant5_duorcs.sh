MODEL_NAME="google/flan-t5-base"

# GreenTrainer-0.5
python3 main.py --model_name $MODEL_NAME \
                --dataset_name duorcs \
                --scheme green_trainer \
                --max_input_length 1024 \
                --max_output_length 64 \
                --batch_size 8 \
                --rho 0.5

# GreenTrainer-0.7
python3 main.py --model_name $MODEL_NAME \
                --dataset_name duorcs \
                --scheme green_trainer \
                --max_input_length 1024 \
                --max_output_length 64 \
                --batch_size 8 \
                --rho 0.7

# Full Finetuning
python3 main.py --model_name $MODEL_NAME \
                --dataset_name duorcs \
                --scheme baselines \
                --train_type full_finetuning \
                --max_input_length 1024 \
                --max_output_length 64 \
                --batch_size 8

# LoRA
python3 main.py --model_name $MODEL_NAME \
                --dataset_name duorcs \
                --scheme baselines \
                --train_type lora \
                --max_input_length 1024 \
                --max_output_length 64 \
                --batch_size 8