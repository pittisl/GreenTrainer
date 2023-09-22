MODEL_NAME="bigscience/bloomz-3b" # bigscience/bloomz-3b

# GreenTrainer-0.5
echo '### GreenTrainer-0.5'
python3 main.py --model_name $MODEL_NAME \
                --dataset_name dialogsum \
                --scheme green_trainer \
                --max_input_length 512 \
                --max_output_length 128 \
                --batch_size 4 \
                --rho 0.5

# GreenTrainer-0.7
echo '### GreenTrainer-0.7'
python3 main.py --model_name $MODEL_NAME \
                --dataset_name dialogsum \
                --scheme green_trainer \
                --max_input_length 512 \
                --max_output_length 128 \
                --batch_size 4 \
                --rho 0.7

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


