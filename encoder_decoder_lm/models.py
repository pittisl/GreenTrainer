from transformers import AutoModelForSeq2SeqLM, AutoConfig
from peft import (
    get_peft_model,
    LoraConfig, AdaLoraConfig, PrefixTuningConfig,
)

def load_text_generation_model(
    model_type, 
    train_type, 
    output_attentions=False,
):
    """Load a seq2seq model with the specified training type.
    
       Options of model_type include:
       `google/flan-t5-small` (80M),
       `google/flan-t5-base` (250M),
       `google/flan-t5-large` (780M),
       `facebook/bart-base` (140M),
       `facebook/bart-large` (406M),  
       `facebook/blenderbot_small-90M` (90M).
       
       Options of train_type include:
       `full_finetuning`, `bifit`, `layer_norm`,
       `lora`, `adalora`, `prefix_tuning`, `teft`,
    """
    
    config = AutoConfig.from_pretrained(
        model_type, output_attentions=output_attentions,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(model_type, config=config)
        
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
            task_type="SEQ_2_SEQ_LM", 
            inference_mode=False, 
            r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.1,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    elif train_type == 'adalora':
        peft_config = AdaLoraConfig(
            peft_type="ADALORA", 
            r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.1,
            task_type="SEQ_2_SEQ_LM",
            inference_mode=False,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
       
    elif train_type == 'prefix_tuning':
        peft_config = PrefixTuningConfig(
            peft_type="PREFIX_TUNING", 
            task_type="SEQ_2_SEQ_LM",
            inference_mode=False, 
            num_virtual_tokens=20,
            num_transformer_submodules=2,
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        
    else:
        raise NotImplementedError(f"{train_type} has not been implemented yet")
    
    return model
    
        