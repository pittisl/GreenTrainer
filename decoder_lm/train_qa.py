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
        self.exact_match = evaluate.load('exact_match')
        self.interval = 200
        self.max_output_length = max_output_length
        self.model_path = model_path

    def train(
        self,
        learning_rate,
        num_epochs,
        log_dir='logs/'
    ):  
        # torch.backends.cuda.matmul.allow_tf32 = True
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=(len(self.train_loader) * num_epochs),
        )
        self.model = self.model.to(self.device)
        writer = SummaryWriter(log_dir=log_dir)
        
        total_time = 0
        # self._runtime_evaluate(self.val_loader)
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
            
            self._runtime_evaluate(self.val_loader)

            # self._save_model()
            
        print(f"Total Time: {total_time} (s)")
        self._runtime_evaluate(self.test_loader)
    
    def _runtime_evaluate(self, dataset):
        self.model.eval()
        # for question answering
        m_em = 0
        self.tokenizer.padding_side = "left"
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataset)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].shape[0]
                
                if self.train_type == "prefix_tuning" or self.train_type == "lora" or self.train_type == "adalora":
                    outputs_tokens = self.model.generate(
                        input_ids=batch['lp_sources'],
                        max_new_tokens=self.max_output_length,
                        num_beams=4,
                        repetition_penalty=2.5, 
                        length_penalty=1.0, 
                        early_stopping=True
                    )
                else:
                    outputs_tokens = self.model.generate(
                        inputs=batch['lp_sources'],
                        max_new_tokens=self.max_output_length,
                        num_beams=4,
                        repetition_penalty=2.5, 
                        length_penalty=1.0, 
                        early_stopping=True
                    )
                
                for label in batch["labels"]:
                    label[label < 0] = self.tokenizer.pad_token_id
                
                outputs_text = [self.tokenizer.decode(y[len(x):], skip_special_tokens=True).strip() for y, x in zip(outputs_tokens, batch['lp_sources'])]
                labels_text =  [self.tokenizer.decode(x[offset:], skip_special_tokens=True).strip() for x, offset in zip(batch["input_ids"], batch["input_ids_lens"])] 
                
                # print(f"pred: {outputs_text[0]} | label: {labels_text[0]}")

                batch_em = self.exact_match.compute(predictions=outputs_text, references=labels_text)
                m_em += (batch_em["exact_match"] * batch_size)
                
                total_count += batch_size
        
        m_em /= total_count
        print(f"On validation/test set, exact_match={100*m_em}")
        
        
        
        # # for summarization
        # m_rouge1 = 0
        # m_rouge2 = 0
        # m_rougeL = 0
        # m_rougeLsum = 0
        
        # total_count = 0
        # with torch.no_grad():
        #     for step, batch in enumerate(tqdm(dataset)):
        #         batch = {k: v.to(self.device) for k, v in batch.items()}
        #         batch_size = batch['input_ids'].shape[0]
                    
        #         all_results = generate_response(
        #             self.model, 
        #             self.train_type,
        #             self.tokenizer, 
        #             batch['lp_sources'], batch['labels'], batch['input_ids_lens'],
        #             max_length=self.max_output_length
        #         )
                
        #         summarization_results = self.rouge_metric.compute(predictions=all_results["outputs_text"], references=all_results["labels_text"])
                
        #         m_rouge1 += (summarization_results['rouge1'] * batch_size)
        #         m_rouge2 += (summarization_results['rouge2'] * batch_size)
        #         m_rougeL += (summarization_results['rougeL'] * batch_size)
        #         m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                
        #         total_count += batch_size
        
        # m_rouge1 /= total_count
        # m_rouge2/= total_count
        # m_rougeL /= total_count
        # m_rougeLsum /= total_count
        # print(f"On validation/test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")

    def evaluate(self):
        self._load_model()
        self.model = self.model.to(self.device)
        
        self.model.eval()
        # for summarization
        m_rouge1 = 0
        m_rouge2 = 0
        m_rougeL = 0
        m_rougeLsum = 0
        
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
                
                m_rouge1 += (summarization_results['rouge1'] * batch_size)
                m_rouge2 += (summarization_results['rouge2'] * batch_size)
                m_rougeL += (summarization_results['rougeL'] * batch_size)
                m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                
                total_count += batch_size
        
        m_rouge1 /= total_count
        m_rouge2/= total_count
        m_rougeL /= total_count
        m_rougeLsum /= total_count
        print(f"On test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")
    
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
        self.exact_match = evaluate.load('exact_match')
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
        # torch.backends.cuda.matmul.allow_tf32 = True
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
            for idx, (_, param) in enumerate(self.model.named_parameters()):
                if idx < N_limit:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            # cache original weight values
            w_0 = [param.data.clone().detach().cpu()  for _, param in self.model.named_parameters()]
            
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
            w_1 = [param.data.clone().detach().cpu() for _, param in self.model.named_parameters()]
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
            grad_1 = [param.grad.clone().detach().cpu() if param.grad is not None else torch.tensor(0.0).cpu() for _, param in self.model.named_parameters()]
            optimizer.step()
            # lr_scheduler.step()
            optimizer.zero_grad()
            I = [torch.sum((grad_1_k * dw_0_k)) for (grad_1_k, dw_0_k) in zip(grad_1, dw_0)]
            I = torch.tensor(I)
            # print(I)
            I = I / torch.max(torch.abs(I))
            # print(I)
            # restore weights
            for k, (_, param) in enumerate(self.model.named_parameters()):
                param.data = w_0[k]
            return I
        
        def to_backward_rho(rho, t_fp, t_dy, t_dw):
            t_bp = np.sum(t_dy + t_dw)
            rho_bp = rho * (1 + t_fp / t_bp) - t_fp / t_bp
            if rho_bp <= 0:
                rho_bp = 0.05
                rho_reset = (rho_bp + t_fp / t_bp) / (1 + t_fp / t_bp)
                print(f"rho is too low. rho has been reset to {rho_reset}")
            return rho_bp
        
        rho_bp = to_backward_rho(rho, t_fp, t_dy, t_dw)
        N = t_dw.shape[0]
        T = np.sum(t_dw + t_dy) # maximally possible BP time
        T_limit = rho_bp * T
        t_dy_cumsum = 0
        t_dy_flipped = np.flip(t_dy)
        for k in range(N):
            t_dy_cumsum += t_dy_flipped[k]
            if t_dy_cumsum > T_limit:
                break
        N_limit = N - k
        print(f"N: {N}, N_limit: {N_limit}")
        
        print(f"T_dy: {np.sum(t_dy)}, {np.sum(t_dy) / np.sum(t_dw + t_dy)}")

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
                I = compute_tensor_importance(batch)
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
                self.model.to(self.device)
            
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
            
            self._runtime_evaluate(self.val_loader)

            # self._save_model()
        
        print(f"Total GreenTrainer Time: {total_time} (s)")
        print(f"Total GreenTrainer Backward GFLOPs: {len(self.train_loader) * np.sum(total_bp_flops):.2f}")
        print(f"Total GreenTrainer GFLOPs: {len(self.train_loader) * np.sum(total_bfp_flops):.2f}")
        print(f"Total GreenTrainer GFLOPs Speedup: {np.mean(total_bfp_speedup):.2f}x")
        
        self._runtime_evaluate(self.test_loader)
        
    def _runtime_evaluate(self, dataset):
        self.model.eval()
        # for question answering
        m_em = 0
        self.tokenizer.padding_side = "left"
        total_count = 0
        with torch.no_grad():
            for step, batch in enumerate(tqdm(dataset)):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                batch_size = batch['input_ids'].shape[0]
                
                if self.train_type == "prefix_tuning" or self.train_type == "lora" or self.train_type == "adalora":
                    outputs_tokens = self.model.generate(
                        input_ids=batch['lp_sources'],
                        max_new_tokens=self.max_output_length,
                        num_beams=4,
                        repetition_penalty=2.5, 
                        length_penalty=1.0, 
                        early_stopping=True
                    )
                else:
                    outputs_tokens = self.model.generate(
                        inputs=batch['lp_sources'],
                        max_new_tokens=self.max_output_length,
                        num_beams=4,
                        repetition_penalty=2.5, 
                        length_penalty=1.0, 
                        early_stopping=True
                    )
                
                for label in batch["labels"]:
                    label[label < 0] = self.tokenizer.pad_token_id
                
                outputs_text = [self.tokenizer.decode(y[len(x):], skip_special_tokens=True).strip() for y, x in zip(outputs_tokens, batch['lp_sources'])]
                labels_text =  [self.tokenizer.decode(x[offset:], skip_special_tokens=True).strip() for x, offset in zip(batch["input_ids"], batch["input_ids_lens"])] 
                
                # print(f"pred: {outputs_text[0]} | label: {labels_text[0]}")

                batch_em = self.exact_match.compute(predictions=outputs_text, references=labels_text)
                m_em += (batch_em["exact_match"] * batch_size)
                
                total_count += batch_size
        
        m_em /= total_count
        print(f"On validation/test set, exact_match={100*m_em}")
        
        
        
        # # for summarization
        # m_rouge1 = 0
        # m_rouge2 = 0
        # m_rougeL = 0
        # m_rougeLsum = 0
        
        # total_count = 0
        # with torch.no_grad():
        #     for step, batch in enumerate(tqdm(dataset)):
        #         batch = {k: v.to(self.device) for k, v in batch.items()}
        #         batch_size = batch['input_ids'].shape[0]
                    
        #         all_results = generate_response(
        #             self.model, 
        #             self.train_type,
        #             self.tokenizer, 
        #             batch['lp_sources'], batch['labels'], batch['input_ids_lens'],
        #             max_length=self.max_output_length
        #         )
                
        #         summarization_results = self.rouge_metric.compute(predictions=all_results["outputs_text"], references=all_results["labels_text"])
                
        #         m_rouge1 += (summarization_results['rouge1'] * batch_size)
        #         m_rouge2 += (summarization_results['rouge2'] * batch_size)
        #         m_rougeL += (summarization_results['rougeL'] * batch_size)
        #         m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                
        #         total_count += batch_size
        
        # m_rouge1 /= total_count
        # m_rouge2/= total_count
        # m_rougeL /= total_count
        # m_rougeLsum /= total_count
        # print(f"On validation/test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")

    def evaluate(self):
        self._load_model()
        self.model = self.model.to(self.device)
        
        self.model.eval()
        # for summarization
        m_rouge1 = 0
        m_rouge2 = 0
        m_rougeL = 0
        m_rougeLsum = 0
        
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
                
                m_rouge1 += (summarization_results['rouge1'] * batch_size)
                m_rouge2 += (summarization_results['rouge2'] * batch_size)
                m_rougeL += (summarization_results['rougeL'] * batch_size)
                m_rougeLsum += (summarization_results['rougeLsum'] * batch_size)
                
                total_count += batch_size
        
        m_rouge1 /= total_count
        m_rouge2/= total_count
        m_rougeL /= total_count
        m_rougeLsum /= total_count
        print(f"On test set, rouge1={100*m_rouge1}, rouge2={100*m_rouge2}, rougeL={100*m_rougeL}, rougeLsum={100*m_rougeLsum}")
    
    def _save_model(self):
        self.model.save_pretrained(self.model_path)
    
    def _load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path)

