import numpy as np
import matplotlib.pyplot as plt


def compute_forward_flops(
    model,
    model_name,
    input_length,
    batch_size=1,
):
    """compute forward flops during training
    
    Supported: 
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
    """
    
    if "opt" in model_name:
        flops = compute_forward_flops_opt(
            model=model,
            input_length=input_length,
            batch_size=batch_size,
        )
    elif "bloom" in model_name:
        flops = compute_forward_flops_bloom(
            model=model,
            input_length=input_length,
            batch_size=batch_size,
        )
    else:
        raise NotImplementedError(f"{model_name} hasn't been implemented yet!")
    
    return flops


def compute_tensor_flops(
    model,
    model_name,
    input_length,
    batch_size=1,
    draw_figure=False,
):
    """compute training flops for NN trainable tensors
    
    Supported: 
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
    """
    
    if "opt" in model_name:
        t_dy, t_dw = compute_tensor_flops_opt(
            model=model,
            input_length=input_length,
            batch_size=batch_size,
            draw_figure=draw_figure,
        )
    elif "bloom" in model_name:
        t_dy, t_dw = compute_tensor_flops_bloom(
            model=model,
            input_length=input_length,
            batch_size=batch_size,
            draw_figure=draw_figure,
        )
    else:
        raise NotImplementedError(f"{model_name} hasn't been implemented yet!")
    
    return t_dy, t_dw


def compute_forward_flops_bloom(
    model,
    input_length,
    batch_size=1,
):
    """Compute forward FLOPs of BLOOM during training"""
    
    flops = 0.0
    
    for param_idx, (name, param) in enumerate(model.named_parameters()):
        if "word_embeddings.weight" in name:
            flops += batch_size * input_length * param.shape[1]
            # output embedding layer's multiplication
            flops += batch_size * input_length * param.shape[0] * param.shape[1]
        
        elif "query_key_value" in name:
            if "weight" in name:
                flops += batch_size * input_length * param.shape[0] * param.shape[1]
            elif "bias" in name:
                flops += batch_size * input_length * param.shape[0]
                attn_map_size = input_length * input_length
                flops += 2 * batch_size * attn_map_size * param.shape[0] // 3
        
        elif "layernorm" in name or "ln_f" in name:
            if "weight" in name:
                flops += batch_size * input_length * param.shape[0]
            elif "bias" in name:
                flops += batch_size * input_length * param.shape[0]
                
        elif "dense" in name:
            if "weight" in name:
                flops += batch_size * input_length * param.shape[0] * param.shape[1]
            elif "bias" in name:
                flops += batch_size * input_length * param.shape[0]
                
    return 1.0 * flops/1e9


def compute_forward_flops_opt(
    model,
    input_length,
    batch_size=1,
):
    """Compute forward FLOPs of OPT during training"""
    
    flops = 0.0
    for param_idx, (name, param) in enumerate(model.named_parameters()):
        if "embed_tokens" in name:
            flops += batch_size * input_length * param.shape[1]
            # output embedding layer's multiplication
            flops += batch_size * input_length * param.shape[0] * param.shape[1]
        elif "embed_positions" in name:
            flops += batch_size * input_length * param.shape[0]
            
        elif ("attn.q" in name) or ("attn.k" in name) or ("attn.o" in name):
            if "weight" in name:
                flops += batch_size * input_length * param.shape[0] * param.shape[1]
            elif "bias" in name:
                flops += batch_size * input_length * param.shape[0]
                
        elif "attn.v" in name:
            if "weight" in name:
                flops += batch_size * input_length * param.shape[0] * param.shape[1]
            elif "bias" in name:
                attn_map_size = input_length * input_length
                flops += batch_size * input_length * param.shape[0]
                flops += 2 * batch_size * attn_map_size * param.shape[0]
        
        elif "layer_norm" in name:
            if "weight" in name:
                flops += batch_size * input_length * param.shape[0]
            elif "bias" in name:
                flops += batch_size * input_length * param.shape[0]
                
        elif "fc" in name:
            if "weight" in name:
                flops += batch_size * input_length * param.shape[0] * param.shape[1]
            elif "bias" in name:
                flops += batch_size * input_length * param.shape[0]
                
    return 1.0 * flops/1e9


def compute_tensor_flops_bloom(
    model,
    input_length,
    batch_size=1,
    draw_figure=False,
):
    """Compute FLOPs of operators that are related to each tensor for BLOOM.
    
    **Note that input and output embedding weights are tied for BLOOM.**
    
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
        if "word_embeddings.weight" in name:
            t_dy[param_idx] = 0
            t_dw[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1] +\
                param.shape[0] * param.shape[1] # t_dw decoder's input embedding layers
            embed_flops = batch_size * input_length * param.shape[0] * param.shape[1]
        
        elif "query_key_value" in name:
            if "weight" in name:
                t_dy[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1]
                t_dw[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = batch_size * input_length * param.shape[0]
                attn_map_size = input_length * input_length
                t_dy[param_idx] += 4 * batch_size * attn_map_size * param.shape[0] // 3
                
        elif "layernorm" in name or "ln_f" in name:
            if "weight" in name:
                t_dy[param_idx] = batch_size * input_length * param.shape[0]
                t_dw[param_idx] = batch_size * input_length * param.shape[0]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = 0
                
        elif "dense" in name:
            if "weight" in name:
                t_dy[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1]
                t_dw[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = batch_size * input_length * param.shape[0]
        # print(f"{name} : {param.shape}")
    t_dy[-1] += embed_flops
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


def compute_tensor_flops_opt(
    model,
    input_length,
    batch_size=1,
    draw_figure=False,
):
    """Compute FLOPs of operators that are related to each tensor for OPT.
    
    **Note that input and output embedding weights are tied for OPT.**
    
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
        if "embed_tokens" in name:
            t_dy[param_idx] = 0
            t_dw[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1] +\
                param.shape[0] * param.shape[1] # t_dw decoder's input embedding layers
            embed_flops = batch_size * input_length * param.shape[0] * param.shape[1]
        
        elif "embed_positions" in name:
            t_dy[param_idx] = 0
            t_dw[param_idx] = param.shape[0]
        
        elif ("attn.q" in name) or ("attn.k" in name) or ("attn.o" in name):
            if "weight" in name:
                t_dy[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1]
                t_dw[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = batch_size * input_length * param.shape[0]
                
        elif "attn.v" in name:
            if "weight" in name:
                t_dy[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1]
                t_dw[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1]
            elif "bias" in name:
                attn_map_size = input_length * input_length
                t_dy[param_idx] = 0
                t_dy[param_idx] += 4 * batch_size * attn_map_size * param.shape[0]
                t_dw[param_idx] = batch_size * input_length * param.shape[0]
                
        elif "layer_norm" in name:
            if "weight" in name:
                t_dy[param_idx] = batch_size * input_length * param.shape[0]
                t_dw[param_idx] = batch_size * input_length * param.shape[0]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = 0
                
        elif "fc" in name:
            if "weight" in name:
                t_dy[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1]
                t_dw[param_idx] = batch_size * input_length * param.shape[0] * param.shape[1]
            elif "bias" in name:
                t_dy[param_idx] = 0
                t_dw[param_idx] = batch_size * input_length * param.shape[0]
        # print(f"{name} : {param.shape}")
    t_dy[-1] += embed_flops
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