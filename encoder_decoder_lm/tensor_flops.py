import numpy as np
import matplotlib.pyplot as plt


def compute_forward_flops(
    model,
    model_name,
    input_length,
    output_length,
    batch_size=1,
):
    """compute forward flops during training
    
    Supported: 
       `google/flan-t5-small` (80M),
       `google/flan-t5-base` (250M),
       `google/flan-t5-large` (780M),
       `facebook/bart-base` (140M),
       `facebook/bart-large` (406M),  
       `facebook/blenderbot_small-90M` (90M).
    """
    
    if "t5" in model_name:
        flops = compute_forward_flops_flant5(
            model=model,
            input_length=input_length,
            output_length=output_length,
            batch_size=batch_size,
        )
    elif "bart" in model_name:
        flops = compute_forward_flops_bart(
            model=model,
            input_length=input_length,
            output_length=output_length,
            batch_size=batch_size,
        )
    elif "blenderbot" in model_name:
        flops = compute_forward_flops_blenderbot(
            model=model,
            input_length=input_length,
            output_length=output_length,
            batch_size=batch_size,
        )
    else:
        raise NotImplementedError(f"{model_name} hasn't been implemented yet!")
    
    return flops


def compute_tensor_flops(
    model,
    model_name,
    input_length,
    output_length,
    batch_size=1,
    draw_figure=False,
):
    """compute training flops for NN trainable tensors
    
    Supported: 
       `google/flan-t5-small` (80M),
       `google/flan-t5-base` (250M),
       `google/flan-t5-large` (780M),
       `facebook/bart-base` (140M),
       `facebook/bart-large` (406M),  
       `facebook/blenderbot_small-90M` (90M).
    """
    
    if "t5" in model_name:
        t_dy, t_dw = compute_tensor_flops_flant5(
            model=model,
            input_length=input_length,
            output_length=output_length,
            batch_size=batch_size,
            draw_figure=draw_figure,
        )
    elif "bart" in model_name:
        t_dy, t_dw = compute_tensor_flops_bart(
            model=model,
            input_length=input_length,
            output_length=output_length,
            batch_size=batch_size,
            draw_figure=draw_figure,
        )
    elif "blenderbot" in model_name:
        t_dy, t_dw = compute_tensor_flops_blenderbot(
            model=model,
            input_length=input_length,
            output_length=output_length,
            batch_size=batch_size,
            draw_figure=draw_figure,
        )
    else:
        raise NotImplementedError(f"{model_name} hasn't been implemented yet!")
    
    return t_dy, t_dw


def compute_forward_flops_flant5(
    model,
    input_length,
    output_length,
    batch_size=1,
):
    """Compute forward FLOPs of FLAN-T5 during training"""
    
    flops = 0.0
    for param_idx, (name, param) in enumerate(model.named_parameters()):
        if "shared" in name:
            flops += batch_size * (input_length + output_length) * param.shape[1]
            
        elif ("Attention.q" in name) or ("Attention.k" in name) or ("Attention.o" in name):
            num_tokens = input_length if "encoder." in name else output_length
            flops += batch_size * num_tokens * param.shape[0] * param.shape[1]
        
        elif "Attention.v" in name:
            num_tokens = input_length if "encoder." in name else output_length
            
            if "EncDec" in name:
                attn_map_size = input_length * output_length
            elif "encoder." in name:
                attn_map_size = input_length * input_length
            else:
                attn_map_size = output_length * output_length
            
            flops += batch_size * num_tokens * param.shape[0] * param.shape[1]
            flops += 2 * batch_size * attn_map_size * param.shape[1]
        
        elif "decoder.final_layer_norm" in name:
            num_tokens = input_length if "encoder." in name else output_length
            flops += batch_size * num_tokens * param.shape[0]
        
        elif "layer_norm" in name:
            num_tokens = input_length if "encoder." in name else output_length
            flops += batch_size * num_tokens * param.shape[0]
        
        elif "DenseReluDense" in name:
            num_tokens = input_length if "encoder." in name else output_length
            flops += batch_size * num_tokens * param.shape[0] * param.shape[1]
        
        elif "lm_head" in name:
            flops += batch_size * output_length * param.shape[0] * param.shape[1]
            
    return 1.0 * flops/1e9
        

def compute_tensor_flops_flant5(
    model,
    input_length,
    output_length,
    batch_size=1,
    draw_figure=False,
):
    """Compute FLOPs of operators that are related to each tensor for FLAN-T5.
    
    **Note that input and output embedding weights are NOT tied for FLAN-T5.**
    https://huggingface.co/google/flan-t5-base/blob/main/config.json#:~:text=%22tie_word_embeddings%22%3A%20false%2C
    
    B: batch size
    D: token dimension
    V: vocabulary size
    N: number of tokens
    F: hidden size of FFN
    
    Rules for supported layers (exclude less significant ops):
    
    `shared`: t_dy = 0, t_dw = 2 * B * N * D * V
    
    `q,k,o`: t_dy = B * N * D * D, t_dw = B * N * D * D
    
    `v`: t_dy = B * N * D * D + 2 * B * N * N * D, t_dw = B * N * D * D
    
    `layer_norm`: t_dy = B * N * D, t_dw = B * N * D
    
    `DenseReluDense.wi`: t_dy = B * N * D * F, t_dw = B * N * D * F
    
    `DenseReluDense.wo`: t_dy = B * N * D * F, t_dw = B * N * D * F
    """
    
    # initialize tensor flops
    t_dy = [0.0 for k in range(len(list(model.named_parameters())))]
    t_dw = [0.0 for k in range(len(list(model.named_parameters())))]
    
    for param_idx, (name, param) in enumerate(model.named_parameters()):
        if "shared" in name:
            t_dy[param_idx] = 0
            # t_dw for both encoder and decoder's input embedding layers
            t_dw[param_idx] = param.shape[0] * param.shape[1] * 2
            
        elif ("Attention.q" in name) or ("Attention.k" in name) or ("Attention.o" in name):
            num_tokens = input_length if "encoder." in name else output_length
            t_dy[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
            t_dw[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
            
        elif "Attention.v" in name:
            num_tokens = input_length if "encoder." in name else output_length
            t_dy[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
            
            if "EncDec" in name:
                attn_map_size = input_length * output_length
            elif "encoder." in name:
                attn_map_size = input_length * input_length
            else:
                attn_map_size = output_length * output_length
                
            t_dy[param_idx] += 4 * batch_size * attn_map_size * param.shape[1]
            t_dw[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
            
        elif "decoder.final_layer_norm" in name:
            num_tokens = input_length if "encoder." in name else output_length
            t_dy[param_idx] = batch_size * num_tokens * param.shape[0]
            t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
            
        elif "layer_norm" in name:
            num_tokens = input_length if "encoder." in name else output_length
            t_dy[param_idx] = batch_size * num_tokens * param.shape[0]
            t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
            
        elif "DenseReluDense" in name:
            num_tokens = input_length if "encoder." in name else output_length
            t_dy[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
            t_dw[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
        
        elif "lm_head" in name:
            t_dy[param_idx] = batch_size * output_length * param.shape[0] * param.shape[1]
            t_dw[param_idx] = batch_size * output_length * param.shape[0] * param.shape[1]
        
        # print(f"({param_idx}) {name} : t_dy={t_dy[param_idx]}, t_dw={t_dw[param_idx]}")
    
    t_dy, t_dw = np.array(t_dy)/1e9, np.array(t_dw)/1e9
    # print(f"{t_dw[0] / np.sum(t_dw + t_dy) * 100} %")
    
    if draw_figure:
        fig = plt.figure(1)
        plt.barh(np.arange(t_dy.shape[0]), t_dy, color ='navy')
        #plt.xticks(rotation=45)
        plt.xlabel('t_dy (GFLOPs)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Tensor ID', fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure(2)
        plt.barh(np.arange(t_dw.shape[0]), t_dw, color ='navy')
        #plt.xticks(rotation=45)
        plt.xlabel('t_dw (GFLOPs)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Tensor ID', fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()
    return t_dy, t_dw


def compute_forward_flops_bart(
    model,
    input_length,
    output_length,
    batch_size=1,
):
    """Compute forward FLOPs of BART during training"""
    
    flops = 0.0
    for param_idx, (name, param) in enumerate(model.named_parameters()):
        if "shared" in name:
            flops += batch_size * (input_length + output_length) * param.shape[1]
            # output embedding layer's multiplication
            flops += batch_size * output_length * param.shape[0] * param.shape[1]
        
        elif "embed_positions" in name:
            flops += batch_size * (input_length + output_length) * param.shape[0]
            
        elif ("attn.q" in name) or ("attn.k" in name) or ("attn.o" in name):
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                flops += batch_size * num_tokens * param.shape[0]
        
        elif "attn.v" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                if "encoder_attn" in name:
                    attn_map_size = input_length * output_length
                elif "encoder." in name:
                    attn_map_size = input_length * input_length
                else:
                    attn_map_size = output_length * output_length
                flops += batch_size * num_tokens * param.shape[0]
                flops += 2 * batch_size * attn_map_size * param.shape[0]
        
        elif "encoder.layernorm_embedding" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                flops += batch_size * num_tokens * param.shape[0]
           
        elif "decoder.layernorm_embedding" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                flops += batch_size * num_tokens * param.shape[0]
                
        elif "layer_norm" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                flops += batch_size * num_tokens * param.shape[0]
                
        elif "fc" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                flops += batch_size * num_tokens * param.shape[0]
                
    return 1.0 * flops/1e9


def compute_tensor_flops_bart(
    model,
    input_length,
    output_length,
    batch_size=1,
    draw_figure=False,
):
    """Compute FLOPs of operators that are related to each tensor for BART.
    
    **Note that input and output embedding weights are tied for BART.**
    
    B: batch size
    D: token dimension
    V: vocabulary size
    N: number of tokens
    F: hidden size of FFN
    
    Rules for supported layers (exclude less significant ops):
    
    `shared`: t_dy = 0, t_dw = B * N * D * V + D * V
    
    `q,k,o`: t_dy = B * N * D * D, t_dw = B * N * D * D
    
    `v`: t_dy = B * N * D * D + 2 * B * N * N * D, t_dw = B * N * D * D
    
    `layer_norm`: t_dy = B * N * D, t_dw = B * N * D
    
    `fc1`: t_dy = B * N * D * F, t_dw = B * N * D * F
    
    `fc2`: t_dy = B * N * D * F, t_dw = B * N * D * F
    """
    
    # initialize tensor flops
    t_dy = [0.0 for k in range(len(list(model.named_parameters())))]
    t_dw = [0.0 for k in range(len(list(model.named_parameters())))]
    embed_flops = 0
    
    for param_idx, (name, param) in enumerate(model.named_parameters()):
        if "shared" in name:
            t_dy[param_idx] = 0
            t_dw[param_idx] = batch_size * output_length * param.shape[0] * param.shape[1] +\
                2 * param.shape[0] * param.shape[1] # t_dw for both encoder and decoder's input embedding layers
            embed_flops = batch_size * output_length * param.shape[0] * param.shape[1]
            
        elif "embed_positions" in name:
            t_dy[param_idx] = 0
            t_dw[param_idx] = param.shape[0]
            
        elif ("attn.q" in name) or ("attn.k" in name) or ("attn.o" in name):
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
                
        elif "attn.v" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                if "encoder_attn" in name:
                    attn_map_size = input_length * output_length
                elif "encoder." in name:
                    attn_map_size = input_length * input_length
                else:
                    attn_map_size = output_length * output_length
                
                t_dy[param_idx] = 0
                t_dy[param_idx] += 4 * batch_size * attn_map_size * param.shape[0]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
        
        elif "encoder.layernorm_embedding" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = 0
           
        elif "decoder.layernorm_embedding" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dy[param_idx] += embed_flops
                t_dw[param_idx] = 0
                
        elif "layer_norm" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = 0
                
        elif "fc" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
        # print(f"{name} : {param.shape}")
    t_dy, t_dw = np.array(t_dy)/1e9, np.array(t_dw)/1e9
    # print(f"{t_dw[0] / np.sum(t_dw + t_dy) * 100} %")
    
    if draw_figure:
        fig = plt.figure(1)
        plt.barh(np.arange(t_dy.shape[0]), t_dy, color ='navy')
        #plt.xticks(rotation=45)
        plt.xlabel('t_dy (GFLOPs)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Tensor ID', fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure(2)
        plt.barh(np.arange(t_dw.shape[0]), t_dw, color ='navy')
        #plt.xticks(rotation=45)
        plt.xlabel('t_dw (GFLOPs)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Tensor ID', fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()
    return t_dy, t_dw


def compute_forward_flops_blenderbot(
    model,
    input_length,
    output_length,
    batch_size=1,
):
    """Compute forward FLOPs of Blenderbot during training"""
    
    flops = 0.0
    for param_idx, (name, param) in enumerate(model.named_parameters()):
        if "shared" in name:
            flops += batch_size * (input_length + output_length) * param.shape[1]
            # output embedding layer's multiplication
            flops += batch_size * output_length * param.shape[0] * param.shape[1]
        
        elif "embed_positions" in name:
            flops += batch_size * (input_length + output_length) * param.shape[0]
            
        elif ("attn.q" in name) or ("attn.k" in name) or ("attn.o" in name):
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                flops += batch_size * num_tokens * param.shape[0]
        
        elif "attn.v" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                if "encoder_attn" in name:
                    attn_map_size = input_length * output_length
                elif "encoder." in name:
                    attn_map_size = input_length * input_length
                else:
                    attn_map_size = output_length * output_length
                flops += batch_size * num_tokens * param.shape[0]
                flops += 2 * batch_size * attn_map_size * param.shape[0]
        
        elif "encoder.layernorm_embedding" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                flops += batch_size * num_tokens * param.shape[0]
           
        elif "decoder.layernorm_embedding" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                flops += batch_size * num_tokens * param.shape[0]
                
        elif "layer_norm" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                flops += batch_size * num_tokens * param.shape[0]
                
        elif "fc" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                flops += batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                flops += batch_size * num_tokens * param.shape[0]
                
    return 1.0 * flops/1e9


def compute_tensor_flops_blenderbot(
    model,
    input_length,
    output_length,
    batch_size=1,
    draw_figure=False,
):
    """Compute FLOPs of operators that are related to each tensor for Blenderbot.
    
    **Note that input and output embedding weights are tied for Blenderbot.**
    
    B: batch size
    D: token dimension
    V: vocabulary size
    N: number of tokens
    F: hidden size of FFN
    
    Rules for supported layers (exclude less significant ops):
    
    `shared`: t_dy = 0, t_dw = B * N * D * V + D * V
    
    `q,k,o`: t_dy = B * N * D * D, t_dw = B * N * D * D
    
    `v`: t_dy = B * N * D * D + 2 * B * N * N * D, t_dw = B * N * D * D
    
    `layer_norm`: t_dy = B * N * D, t_dw = B * N * D
    
    `fc1`: t_dy = B * N * D * F, t_dw = B * N * D * F
    
    `fc2`: t_dy = B * N * D * F, t_dw = B * N * D * F
    """
    
    # initialize tensor flops
    t_dy = [0.0 for k in range(len(list(model.named_parameters())))]
    t_dw = [0.0 for k in range(len(list(model.named_parameters())))]
    embed_flops = 0
    
    for param_idx, (name, param) in enumerate(model.named_parameters()):
        if "shared" in name:
            t_dy[param_idx] = 0
            t_dw[param_idx] = batch_size * output_length * param.shape[0] * param.shape[1] +\
                2 * param.shape[0] * param.shape[1] # t_dw for both encoder and decoder's input embedding layers
            embed_flops = batch_size * output_length * param.shape[0] * param.shape[1]
            
        elif "embed_positions" in name:
            t_dy[param_idx] = 0
            t_dw[param_idx] = param.shape[0]
            
        elif ("attn.q" in name) or ("attn.k" in name) or ("attn.o" in name):
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
                
        elif "attn.v" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                if "encoder_attn" in name:
                    attn_map_size = input_length * output_length
                elif "encoder." in name:
                    attn_map_size = input_length * input_length
                else:
                    attn_map_size = output_length * output_length
                
                t_dy[param_idx] = 0
                t_dy[param_idx] += 4 * batch_size * attn_map_size * param.shape[0]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
        
        elif "encoder.layernorm_embedding" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = 0
        
        elif "decoder.layernorm_embedding" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dy[param_idx] += embed_flops
                t_dw[param_idx] = 0
                
        elif "layer_norm" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = 0
                
        elif "fc" in name:
            num_tokens = input_length if "encoder." in name else output_length
            if "weight" in name:
                t_dy[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0] * param.shape[1]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = batch_size * num_tokens * param.shape[0]
        # print(f"{name} : {param.shape}")
    t_dy, t_dw = np.array(t_dy)/1e9, np.array(t_dw)/1e9
    # print(f"{t_dw[0] / np.sum(t_dw + t_dy) * 100} %")
    
    if draw_figure:
        fig = plt.figure(1)
        plt.barh(np.arange(t_dy.shape[0]), t_dy, color ='navy')
        #plt.xticks(rotation=45)
        plt.xlabel('t_dy (GFLOPs)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Tensor ID', fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()
        
        fig = plt.figure(2)
        plt.barh(np.arange(t_dw.shape[0]), t_dw, color ='navy')
        #plt.xticks(rotation=45)
        plt.xlabel('t_dw (GFLOPs)', fontsize=20)
        plt.xticks(fontsize=20)
        plt.ylabel('Tensor ID', fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.show()
    return t_dy, t_dw