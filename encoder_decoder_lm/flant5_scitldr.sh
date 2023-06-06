MODEL_NAME="google/flan-t5-xl"

# GreenTrainer-0.5
python3 main.py --model_name $MODEL_NAME \
                --dataset_name scitldr \
                --scheme green_trainer \
                --max_input_length 512 \
                --max_output_length 64 \
                --batch_size 4 \
                --rho 0.5

# GreenTrainer-0.7
python3 main.py --model_name $MODEL_NAME \
                --dataset_name scitldr \
                --scheme green_trainer \
                --max_input_length 512 \
                --max_output_length 64 \
                --batch_size 4 \
                --rho 0.7

# Full Finetuning
python3 main.py --model_name $MODEL_NAME \
                --dataset_name scitldr \
                --scheme baselines \
                --train_type full_finetuning \
                --max_input_length 512 \
                --max_output_length 64 \
                --batch_size 4

# LoRA
python3 main.py --model_name $MODEL_NAME \
                --dataset_name scitldr \
                --scheme baselines \
                --train_type lora \
                --max_input_length 512 \
                --max_output_length 64 \
                --batch_size 4