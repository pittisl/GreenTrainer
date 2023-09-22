MODEL_NAME="google/flan-t5-xl"

# GreenTrainer-0.34
echo '### GreenTrainer-0.34'
python3 main.py --model_name $MODEL_NAME \
                --dataset_name dialogsum \
                --scheme green_trainer \
                --max_input_length 512 \
                --max_output_length 128 \
                --batch_size 4 \
                --rho 0.34

# GreenTrainer-0.4
echo '### GreenTrainer-0.4'
python3 main.py --model_name $MODEL_NAME \
                --dataset_name dialogsum \
                --scheme green_trainer \
                --max_input_length 512 \
                --max_output_length 128 \
                --batch_size 4 \
                --rho 0.4

# Full Finetuning
echo '### Full Finetuning'
python3 main.py --model_name $MODEL_NAME \
                --dataset_name dialogsum \
                --scheme baselines \
                --train_type full_finetuning \
                --max_input_length 512 \
                --max_output_length 128 \
                --batch_size 4

# LoRA
echo '### LoRA'
python3 main.py --model_name $MODEL_NAME \
                --dataset_name dialogsum \
                --scheme baselines \
                --train_type lora \
                --max_input_length 512 \
                --max_output_length 128 \
                --batch_size 4