import torch
import evaluate
import time
from peft import PeftModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup, AutoModelForCausalLM
from utils import generate_response
import numpy as np
from tensor_selector import selection_DP, downscale_t_dy_and_t_dw
from tensor_flops import compute_tensor_flops, compute_forward_flops
from utils import flops_counter, compute_squad_metric


class Trainer:
    def __init__(
        self, 
        train_loader, 
        val_loader,
        test_loader, 
        model,
        train_type,
        tokenizer,
        max_output_length,
        model_path,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.train_type = train_type
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = SummaryWriter(flush_secs=10)
        self.rouge_metric = evaluate.load('rouge')
        self.interval = 200
        self.max_output_length = max_output_length
        self.model_path = model_path

    def train(
        self,
        learning_rate,
        num_epochs,
        log_dir='logs/'
    ):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_loader) * num_epochs),
        )
        self.model = self.model.to(self.device)
        writer = SummaryWriter(log_dir=log_dir)
        
        total_time = 0
        self._runtime_evaluate()
        for epoch in range(num_epochs):
            t_start = time.time()
            
            self.model.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(self.train_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                # print(batch)
                # print(batch["input_ids"].shape)
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            t_end = time.time()
            epoch_time = t_end - t_start
            print(f"Epoch Time: {epoch_time} (s)")
            total_time += epoch_time
            print(f"Total Time: {total_time} (s)")
            
            self.model.eval()
            eval_loss = 0
            for step, batch in enumerate(tqdm(self.val_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                loss = outputs.loss
                eval_loss += loss.detach().float()
            
            eval_epoch_loss = eval_loss / len(self.val_loader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(self.train_loader)
            train_ppl = torch.exp(train_epoch_loss)
            
            writer.add_scalar('Loss/train', train_epoch_loss.item(), epoch)
            writer.add_scalar('PPL/train', train_ppl.item(), epoch)
            writer.add_scalar('Loss/valid', eval_epoch_loss.item(), epoch)
            writer.add_scalar('PPL/valid', eval_ppl.item(), epoch)
            
            print(f"epoch={epoch} train_ppl={train_ppl.item()} train_loss={train_epoch_loss.item()} eval_ppl={eval_ppl.item()} eval_loss={eval_epoch_loss.item()}")
            
            self._runtime_evaluate()

            self._save_model()
            
        print(f"Total Time: {total_time} (s)")
    
    def _runtime_evaluate(self):
        self.model.eval()
        # for summarization
        m_rouge1 = 0
        m_rouge2 = 0
        m_rougeL = 0
        m_rougeLsum = 0
        # for question answering
        m_f1 = 0
        m_em = 0
        
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.val_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].shape[0]
                    
                all_results = generate_response(
                    self.model, 
                    self.train_type,
                    self.tokenizer, 
                    batch['lp_sources'], batch['labels'], batch['input_ids_lens'],
                    max_length=self.max_output_length
                )
                
                summarization_results = self.rouge_metric.compute(predictions=all_results["outputs_text"], references=all_results["labels_text"])
                qa_results = compute_squad_metric(self.tokenizer, predictions=all_results["outputs_tokens"], references=all_results["labels_tokens"])
                
                m_rouge1 += (summarization_results['rouge1'] * batch_size)
                m_rouge2 += (summarization_results['rouge2'] * batch_size)
                m_rougeL += (summarization_results['rougeL'] * batch_size)
                m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                m_f1 += (qa_results["f1"] * batch_size)
                m_em += (qa_results["EM"] * batch_size)
                
                total_count += batch_size
        
        m_rouge1 /= total_count
        m_rouge2/= total_count
        m_rougeL /= total_count
        m_rougeLsum /= total_count
        m_f1 /= total_count
        m_em /= total_count
        print(f"On validation set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")
        print(f"On validation set, f1={m_f1}%, EM={m_em}%")

    def evaluate(self):
        self._load_model()
        self.model = self.model.to(self.device)
        
        self.model.eval()
        # for summarization
        m_rouge1 = 0
        m_rouge2 = 0
        m_rougeL = 0
        m_rougeLsum = 0
        # for question answering
        m_f1 = 0
        m_em = 0
        
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].shape[0]
                    
                all_results = generate_response(
                    self.model, 
                    self.train_type,
                    self.tokenizer, 
                    batch['lp_sources'], batch['labels'], batch['input_ids_lens'],
                    max_length=self.max_output_length
                )
                
                summarization_results = self.rouge_metric.compute(predictions=all_results["outputs_text"], references=all_results["labels_text"])
                qa_results = compute_squad_metric(self.tokenizer, predictions=all_results["outputs_tokens"], references=all_results["labels_tokens"])
                
                m_rouge1 += (summarization_results['rouge1'] * batch_size)
                m_rouge2 += (summarization_results['rouge2'] * batch_size)
                m_rougeL += (summarization_results['rougeL'] * batch_size)
                m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                m_f1 += (qa_results["f1"] * batch_size)
                m_em += (qa_results["EM"] * batch_size)
                
                total_count += batch_size
        
        m_rouge1 /= total_count
        m_rouge2/= total_count
        m_rougeL /= total_count
        m_rougeLsum /= total_count
        m_f1 /= total_count
        m_em /= total_count
        print(f"On test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")
        print(f"On test set, f1={m_f1}%, EM={m_em}%")
    
    def _save_model(self):
        self.model.save_pretrained(self.model_path)
    
    def _load_model(self):
        if self.train_type in ["lora", "adalora", "prefix_tuning"]:
            self.model = PeftModel.from_pretrained(self.model, self.model_path)
            # if self.train_type == "lora" or self.train_type == "adalora":
            #     self.model.merge_and_unload()
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path)


class Green_Trainer:
    def __init__(
        self, 
        train_loader, 
        val_loader,
        test_loader, 
        model,
        model_type,
        train_type,
        tokenizer,
        max_output_length,
        model_path,
    ) -> None:
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = model
        self.model_type = model_type
        self.train_type = train_type
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = SummaryWriter(flush_secs=10)
        self.rouge_metric = evaluate.load('rouge')
        self.interval = 200
        self.max_output_length = max_output_length
        self.model_path = model_path

    def train(
        self,
        learning_rate,
        num_epochs,
        interval=2,
        rho=0.3,
        input_length=1024,
        output_length=256,
        batch_size=16,
        log_dir='logs/'
    ):
        t_dy, t_dw = compute_tensor_flops(
            model=self.model,
            model_name=self.model_type,
            input_length=input_length,
            # output_length=output_length,
            batch_size=batch_size,
            draw_figure=False,
        )
        t_fp = compute_forward_flops(
            model=self.model,
            model_name=self.model_type,
            input_length=input_length,
            # output_length=output_length,
            batch_size=batch_size,
        )
        
        t_dy_q, t_dw_q, disco = downscale_t_dy_and_t_dw(t_dy, t_dw, Tq=1e3)
        t_dy_q = np.flip(t_dy_q)
        t_dw_q = np.flip(t_dw_q)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_loader) * num_epochs),
        )
        self.model = self.model.to(self.device)
        writer = SummaryWriter(log_dir=log_dir)
        
        def compute_tensor_importance(batch):
            # set all params to be trainable
            for _, param in self.model.named_parameters():
                param.requires_grad = True
            # cache original weight values
            w_0 = [param.data.clone().detach() for _, param in self.model.named_parameters()]
            
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            # perform update
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            # cache updated weight values
            w_1 = [param.data.clone().detach() for _, param in self.model.named_parameters()]
            # compute weight changes, it takes optimizer's schedule into account
            dw_0 = [w_1_k - w_0_k for (w_0_k, w_1_k) in zip(w_0, w_1)]
            
            # cache gradients
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss.backward()
            grad_1 = [param.grad.clone().detach() for _, param in self.model.named_parameters()]
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            I = [torch.sum((grad_1_k * dw_0_k)) for (grad_1_k, dw_0_k) in zip(grad_1, dw_0)]
            I = torch.tensor(I)
            I = I / torch.max(torch.abs(I))
            # restore weights
            for k, (_, param) in enumerate(self.model.named_parameters()):
                param.data = w_0[k]
            return dw_0, I
        
        def to_backward_rho(rho, t_fp, t_dy, t_dw):
            t_bp = np.sum(t_dy + t_dw)
            rho_bp = rho * (1 + t_fp / t_bp) - t_fp / t_bp
            if rho_bp <= 0:
                rho_bp = 0.05
                rho_reset = (rho_bp + t_fp / t_bp) / (1 + t_fp / t_bp)
                print(f"rho is too low. rho has been reset to {rho_reset}")
            return rho_bp
        
        rho_bp = to_backward_rho(rho, t_fp, t_dy, t_dw)
          
        total_time = 0
        total_bp_flops = []
        total_bfp_flops = []
        total_bfp_speedup = []
        
        for epoch in range(num_epochs):
            t_start = time.time()
            
            self.model.train()
            total_loss = 0
            
            if epoch % interval == 0:
                print("#### Selecting trainable tensors...")
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
                dw, I = compute_tensor_importance(batch)
                I = -I.numpy()
                I = np.flip(I)
                # print("disco:", disco)
                max_importance, m = selection_DP(t_dy_q, t_dw_q, I, rho=rho_bp)
                m = np.flip(m)
                print("m:", m)
                print("max importance:", max_importance)
                print("%T_sel:", 100 * np.sum(np.maximum.accumulate(m) * t_dy + m * t_dw) / np.sum(t_dy + t_dw))
                # ground trainability
                for k, (_, param) in enumerate(self.model.named_parameters()):
                    if m[k] == 1:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            
            for step, batch in enumerate(tqdm(self.train_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss
                total_loss += loss.detach().float()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            flops = flops_counter(self.model, t_fp=t_fp, t_dy=t_dy, t_dw=t_dw)
            t_end = time.time()
            epoch_time = t_end - t_start
            print(f"Epoch Time: {epoch_time} (s)")
            total_time += epoch_time
            print(f"Total Time: {total_time} (s)")
            total_bp_flops.append(flops["bp"])
            total_bfp_flops.append(flops["bfp"])
            total_bfp_speedup.append(flops["full_bfp"] / flops["bfp"])
        
            self.model.eval()
            eval_loss = 0
            for step, batch in enumerate(tqdm(self.val_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                loss = outputs.loss
                eval_loss += loss.detach().float()
            
            eval_epoch_loss = eval_loss / len(self.val_loader)
            eval_ppl = torch.exp(eval_epoch_loss)
            train_epoch_loss = total_loss / len(self.train_loader)
            train_ppl = torch.exp(train_epoch_loss)
            
            writer.add_scalar('Loss/train', train_epoch_loss.item(), epoch)
            writer.add_scalar('PPL/train', train_ppl.item(), epoch)
            writer.add_scalar('Loss/valid', eval_epoch_loss.item(), epoch)
            writer.add_scalar('PPL/valid', eval_ppl.item(), epoch)
            
            print(f"epoch={epoch} train_ppl={train_ppl.item()} train_loss={train_epoch_loss.item()} eval_ppl={eval_ppl.item()} eval_loss={eval_epoch_loss.item()}")
            
            self._runtime_evaluate()

            self._save_model()
        
        print(f"Total GreenTrainer Time: {total_time} (s)")
        print(f"Total GreenTrainer Backward GFLOPs: {len(self.train_loader) * np.sum(total_bp_flops):.2f}")
        print(f"Total GreenTrainer GFLOPs: {len(self.train_loader) * np.sum(total_bfp_flops):.2f}")
        print(f"Total GreenTrainer GFLOPs Speedup: {np.mean(total_bfp_speedup):.2f}x")
    
    
    def _runtime_evaluate(self):
        self.model.eval()
        # for summarization
        m_rouge1 = 0
        m_rouge2 = 0
        m_rougeL = 0
        m_rougeLsum = 0
        # for question answering
        m_f1 = 0
        m_em = 0
        
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.val_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].shape[0]
                    
                all_results = generate_response(
                    self.model, 
                    self.train_type,
                    self.tokenizer, 
                    batch['lp_sources'], batch['labels'], batch['input_ids_lens'],
                    max_length=self.max_output_length
                )
                
                summarization_results = self.rouge_metric.compute(predictions=all_results["outputs_text"], references=all_results["labels_text"])
                qa_results = compute_squad_metric(self.tokenizer, predictions=all_results["outputs_tokens"], references=all_results["labels_tokens"])
                
                m_rouge1 += (summarization_results['rouge1'] * batch_size)
                m_rouge2 += (summarization_results['rouge2'] * batch_size)
                m_rougeL += (summarization_results['rougeL'] * batch_size)
                m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                m_f1 += (qa_results["f1"] * batch_size)
                m_em += (qa_results["EM"] * batch_size)
                
                total_count += batch_size
        
        m_rouge1 /= total_count
        m_rouge2/= total_count
        m_rougeL /= total_count
        m_rougeLsum /= total_count
        m_f1 /= total_count
        m_em /= total_count
        print(f"On validation set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")
        print(f"On validation set, f1={m_f1}%, EM={m_em}%")

    def evaluate(self):
        self._load_model()
        self.model = self.model.to(self.device)
        
        self.model.eval()
        # for summarization
        m_rouge1 = 0
        m_rouge2 = 0
        m_rougeL = 0
        m_rougeLsum = 0
        # for question answering
        m_f1 = 0
        m_em = 0
        
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(self.test_loader)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].shape[0]
                    
                all_results = generate_response(
                    self.model, 
                    self.train_type,
                    self.tokenizer, 
                    batch['lp_sources'], batch['labels'], batch['input_ids_lens'],
                    max_length=self.max_output_length
                )
                
                summarization_results = self.rouge_metric.compute(predictions=all_results["outputs_text"], references=all_results["labels_text"])
                qa_results = compute_squad_metric(self.tokenizer, predictions=all_results["outputs_tokens"], references=all_results["labels_tokens"])
                
                m_rouge1 += (summarization_results['rouge1'] * batch_size)
                m_rouge2 += (summarization_results['rouge2'] * batch_size)
                m_rougeL += (summarization_results['rougeL'] * batch_size)
                m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                m_f1 += (qa_results["f1"] * batch_size)
                m_em += (qa_results["EM"] * batch_size)
                
                total_count += batch_size
        
        m_rouge1 /= total_count
        m_rouge2/= total_count
        m_rougeL /= total_count
        m_rougeLsum /= total_count
        m_f1 /= total_count
        m_em /= total_count
        print(f"On test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")
        print(f"On test set, f1={m_f1}%, EM={m_em}%")
    
    def _save_model(self):
        self.model.save_pretrained(self.model_path)
    
    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

