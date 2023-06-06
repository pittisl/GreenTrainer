from transformers import AutoModelForCausalLM, AutoConfig
from peft import (
    get_peft_model,
    LoraConfig, AdaLoraConfig, PrefixTuningConfig,
)
import torch

def load_text_generation_model(
    model_type, 
    train_type, 
    output_attentions=False,
):
    """Load a seq2seq model with the specified training type.
    
       Options of model_type include:
       `gpt2 (124M)`,
       `gpt2-medium` (355M),
       `gpt2-large` (774M),
       `gpt2-xl` (1.5B),
       `facebook/opt-125m` (125M),
       `facebook/opt-350m` (350M),
       `facebook/opt-1.3b` (1.3B),
       `facebook/opt-2.7b` (2.7B),
       `facebook/opt-6.7b` (6.7B),
       `facebook/opt-13b` (13B),  
       `decapoda-research/llama-7b-hf` (7B).
       
       Options of train_type include:
       `full_finetuning`, `bifit`, `layer_norm`,
       `lora`, `adalora`, `prefix_tuning`, `green_trainer`,
    """
    
    config = AutoConfig.from_pretrained(
        model_type, output_attentions=output_attentions,
    )
    model = AutoModelForCausalLM.from_pretrained(model_type, config=config)
        
    if train_type == 'full_finetuning' or train_type == 'green_trainer':
        return model
    
    elif train_type == 'bitfit':
        for name, param in model.named_parameters():
            if 'bias' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
                
    elif train_type == 'layer_norm':
        for name, param in model.named_parameters():
            if 'layer_norm' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
    elif train_type == 'lora':
        peft_config = LoraConfig(
            peft_type="LORA", 
            task_type="CAUSAL_LM", 
            inference_mode=False, 
            r=8, lora_alpha=32, target_modules=["q_proj", "v_proj"], # ["q_proj", "v_proj"] for opt, ["query_key_value"] for bloom
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    elif train_type == 'adalora':
        peft_config = AdaLoraConfig(
            peft_type="ADALORA", 
            r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
            inference_mode=False,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
       
    elif train_type == 'prefix_tuning':
        peft_config = PrefixTuningConfig(
            peft_type="PREFIX_TUNING", 
            task_type="CAUSAL_LM",
            inference_mode=False, 
            num_virtual_tokens=20,
            num_transformer_submodules=1,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    else:
        raise NotImplementedError(f"{train_type} has not been implemented yet")
    
    return model
    
        
