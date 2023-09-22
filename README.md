# Introduction
This is the official code repository for GreenTrainer. **We are still finalizing the code and please stay tuned for the final release**.
# Requirements
All the experiments are run on Lambda Cloud Instances. To install all the packages, run the following command
```
bash requirements.sh
```
# Basic Usage
For decoder structures, navigate to `decoder_lm` folder. Run the following commands to train

```
bash opt_scitldr.sh # train opt model on scitldr dataset
bash opt_dialogsum.sh # train opt model on scitldr dataset
```

or pass specific configurations to main.py

```
# GreenTrainer-0.5
python3 main.py --model_name $MODEL_NAME \
                --dataset_name scitldr \
                --scheme green_trainer \
                --max_input_length 512 \
                --max_output_length 64 \
                --batch_size 4 \
                --rho 0.4
```

For encoder-decoder structures, navigate to `encoder_decoder_lm`. Follow similar steps above.
