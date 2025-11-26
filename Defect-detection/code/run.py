# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function
from sklearn.model_selection import train_test_split

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    RandomSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
import json

# Import threshold optimization methods
from threshold_optimization import (
    grid_search_threshold,
    ghost_threshold_optimization,
    compare_threshold_methods,
    calculate_metrics_with_threshold
)

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model
import wandb

cpu_cont = multiprocessing.cpu_count()
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    get_linear_schedule_with_warmup,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    BertForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    T5Config,
    T5EncoderModel, 
    T5Tokenizer,
    T5ForConditionalGeneration,
)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "distilbert": (
        DistilBertConfig,
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
    ),

    "codebert-base": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "linevul": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "graphcodebert-base": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "codet5": (T5Config, T5EncoderModel, RobertaTokenizer),
    "codet5_full": (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
    #"natgen": (T5Config, T5EncoderModel, RobertaTokenizer),
    "natgen": (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
}


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(
        self,
        input_tokens,
        input_ids,
        idx,
        label,
    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label


def convert_examples_to_features(js, tokenizer, args):
    # source
    code = " ".join(js["func"].split())
    code_tokens = tokenizer.tokenize(code)[: args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return InputFeatures(source_tokens, source_ids, js["idx"], js["target"])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        with open(file_path) as f:
            for line in f:
                js = json.loads(line.strip())
                self.examples.append(convert_examples_to_features(js, tokenizer, args))
        if "train" in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("label: {}".format(example.label))
                logger.info(
                    "input_tokens: {}".format(
                        [x.replace("\u0120", "_") for x in example.input_tokens]
                    )
                )
                logger.info(
                    "input_ids: {}".format(" ".join(map(str, example.input_ids)))
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(
            self.examples[i].label
        )


def set_seed(seed=42):
    random.seed(seed)
    os.environ["PYHTONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

no_train = False
def train(args, train_dataset, model, tokenizer, tb_writer=None):
    """Train the model"""
    global no_train
    if no_train:
        print("ensuring no training")
        exit(1)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )

    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        num_workers=4,
        pin_memory=True,
    )
    args.max_steps = args.epoch * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = len(train_dataloader)
    args.logging_steps = len(train_dataloader)
    args.num_train_epochs = args.epoch
    model.to(args.device)
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.max_steps * 0.1,
        num_training_steps=args.max_steps,
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    checkpoint_last = os.path.join(args.output_dir, "checkpoint-last")
    scheduler_last = os.path.join(checkpoint_last, "scheduler.pt")
    optimizer_last = os.path.join(checkpoint_last, "optimizer.pt")
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last))
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info(
        "  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size
    )
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", args.max_steps)
    training_log_file = os.path.join(args.output_dir, "training_log.csv")
    with open(training_log_file, "w") as f:
        f.write("epoch,step,train_loss,eval_acc,eval_f1,eval_precision,eval_recall,lr\n")

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    best_acc = 0.0
    model.zero_grad()

    early_stopping_counter = 0
    best_loss = None

    global_wandb_step = 0 

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        
        for step, batch in enumerate(bar):
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            model.train()
            loss, logits = model(inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm
                )
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))
            
            # Log to wandb every 50 steps
            if args.use_wandb and args.local_rank in [-1, 0] and step % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                wandb.log({
                    "train/loss": avg_loss,
                    "train/learning_rate": current_lr,
                    "train/epoch": idx,
                    "train/step": step,
                    "train/global_step": global_wandb_step
                }, step=global_wandb_step)

            # Log to tensorboard every 50 steps
            if tb_writer is not None and args.local_rank in [-1, 0] and step % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                tb_writer.add_scalar("train/loss", avg_loss, global_wandb_step)
                tb_writer.add_scalar("train/learning_rate", current_lr, global_wandb_step)
                tb_writer.add_scalar("train/epoch", idx, global_wandb_step)
                tb_writer.add_scalar("train/step", step, global_wandb_step)
            
            # Add logging every 100 steps (existing code)
            if step % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                # Log to CSV (existing)
                with open(training_log_file, "a") as f:
                    f.write(f"{idx},{step},{avg_loss:.6f},,,,{current_lr:.8f}\n")
                logger.info(f"Epoch {idx}, Step {step}/{len(train_dataloader)}: "
                           f"Loss={avg_loss:.4f}, LR={current_lr:.2e}")

            # IMPORTANT: Only step optimizer every gradient_accumulation_steps
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                global_wandb_step += 1
                
                # Calculate average loss for logging
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4
                )
                
                # Evaluation and model saving (only when we actually step)
                if (
                    args.local_rank in [-1, 0]
                    and args.save_steps > 0
                    and global_step % args.save_steps == 0
                ):
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):
                        results = evaluate(args, model, tokenizer, eval_when_training=True)

                        # Log evaluation results to wandb
                        if args.use_wandb:
                            wandb_metrics = {
                                f"eval/{key}": value for key, value in results.items()
                            }
                            wandb_metrics.update({
                                "eval/epoch": idx,
                                "eval/global_step": global_step,
                            })
                            wandb.log(wandb_metrics, step=global_wandb_step)

                        # Log evaluation results to tensorboard
                        if tb_writer is not None:
                            for key, value in results.items():
                                tb_writer.add_scalar(f"eval/{key}", value, global_wandb_step)
                            tb_writer.add_scalar("eval/epoch", idx, global_wandb_step)
                            tb_writer.add_scalar("eval/global_step", global_step, global_wandb_step)

                        # Log evaluation results (existing code)
                        eval_acc = results.get("eval_acc", 0)
                        eval_f1 = results.get("eval_f1", 0)
                        eval_precision = results.get("eval_precision", 0)
                        eval_recall = results.get("eval_recall", 0)
                        current_lr = optimizer.param_groups[0]['lr']

                        # Log to CSV (existing)
                        with open(training_log_file, "a") as f:
                            f.write(f"{idx},{step},{avg_loss:.6f},{eval_acc:.4f},"
                                   f"{eval_f1:.4f},{eval_precision:.4f},{eval_recall:.4f},{current_lr:.8f}\n")

                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value, 4))

                        # Save model checkpoint if best
                        if results["eval_acc"] > best_acc:
                            best_acc = results["eval_acc"]
                            
                            # Log best model to wandb
                            if args.use_wandb:
                                wandb.log({
                                    "best/accuracy": best_acc,
                                    "best/epoch": idx,
                                    "best/global_step": global_step
                                }, step=global_wandb_step)
                            
                            logger.info("  " + "*" * 20)
                            logger.info("  Best acc:%s", round(best_acc, 4))
                            logger.info("  " + "*" * 20)

                            checkpoint_prefix = "checkpoint-best-acc"
                            checkpoint_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))
                            if not os.path.exists(checkpoint_dir):
                                os.makedirs(checkpoint_dir)
                            model_to_save = (
                                model.module if hasattr(model, "module") else model
                            )

                            # Save model weights
                            model_path = os.path.join(checkpoint_dir, "model.bin")
                            torch.save(model_to_save.state_dict(), model_path)
                            logger.info("Saving model checkpoint to %s", model_path)

                            # Save config.json
                            # Get config from the model (wrapped models store it as model.config)
                            if hasattr(model_to_save, 'config'):
                                config_path = os.path.join(checkpoint_dir, "config.json")
                                model_to_save.config.save_pretrained(checkpoint_dir)
                                logger.info("Saving model config to %s", config_path)

        # END OF STEP LOOP - Log epoch metrics to wandb
        if args.use_wandb and args.local_rank in [-1, 0]:
            wandb.log({
                "epoch/avg_loss": avg_loss,
                "epoch/examples_processed": tr_num * args.train_batch_size,
                "epoch/epoch": idx,
                "epoch/best_acc": best_acc
            }, step=global_wandb_step)

        # Log epoch metrics to tensorboard
        if tb_writer is not None and args.local_rank in [-1, 0]:
            tb_writer.add_scalar("epoch/avg_loss", avg_loss, global_wandb_step)
            tb_writer.add_scalar("epoch/examples_processed", tr_num * args.train_batch_size, global_wandb_step)
            tb_writer.add_scalar("epoch/epoch", idx, global_wandb_step)
            tb_writer.add_scalar("epoch/best_acc", best_acc, global_wandb_step)
        
        # Calculate average loss for the epoch (existing code)
        avg_loss = train_loss / tr_num

        # Add epoch metrics logging
        epoch_metrics = {
            "epoch": idx,
            "avg_loss": avg_loss,
            "examples_processed": tr_num * args.train_batch_size,
            "learning_rate": optimizer.param_groups[0]['lr']
        }

        logger.info("=== End of Epoch {} ===".format(idx))
        for key, value in epoch_metrics.items():
            logger.info(f"  {key}: {value}")

        # Save epoch summary
        epoch_summary_file = os.path.join(args.output_dir, "epoch_summary.txt")
        with open(epoch_summary_file, "a") as f:
            if idx == 0:  # Write header on first epoch
                f.write("Epoch\tAvg_Loss\tBest_Acc\tLR\tExamples\n")
            f.write(f"{idx}\t{avg_loss:.6f}\t{best_acc:.4f}\t"
                   f"{optimizer.param_groups[0]['lr']:.2e}\t{tr_num * args.train_batch_size}\n")

        # Check for early stopping condition (AFTER epoch is complete)
        if args.early_stopping_patience is not None:
            if best_loss is None or avg_loss < best_loss - args.min_loss_delta:
                best_loss = avg_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= args.early_stopping_patience:
                    logger.info("Early stopping")
                    break  # Exit the epoch loop early

# END OF EPOCH LOOP
def evaluate(args, model, tokenizer, eval_when_training=False):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = (
        SequentialSampler(eval_dataset)
        if args.local_rank == -1
        else DistributedSampler(eval_dataset)
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        num_workers=4,
        pin_memory=True,
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    labels = []
    for batch in eval_dataloader:
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            lm_loss, logit = model(inputs, label)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1
    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    # Calculate predictions with different thresholds
    preds_05 = logits[:, 0] > 0.5
    
    # Calculate detailed metrics
    true_pos = ((preds_05 == 1) & (labels == 1)).sum()
    false_pos = ((preds_05 == 1) & (labels == 0)).sum() 
    false_neg = ((preds_05 == 0) & (labels == 1)).sum()
    true_neg = ((preds_05 == 0) & (labels == 0)).sum()
    
    eval_acc = (true_pos + true_neg) / len(labels)
    eval_precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    eval_recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
    eval_f1 = 2 * eval_precision * eval_recall / (eval_precision + eval_recall) if (eval_precision + eval_recall) > 0 else 0
    
    # Calculate AUC (threshold-independent metric)
    from sklearn.metrics import roc_auc_score, average_precision_score
    try:
        eval_auc = roc_auc_score(labels, logits[:, 0])
        eval_ap = average_precision_score(labels, logits[:, 0])
    except:
        eval_auc = 0.0
        eval_ap = 0.0
    
    result = {
        "eval_loss": float(eval_loss / nb_eval_steps),
        "eval_acc": float(eval_acc),
        "eval_precision": float(eval_precision),
        "eval_recall": float(eval_recall), 
        "eval_f1": float(eval_f1),
        "eval_auc": float(eval_auc),
        "eval_ap": float(eval_ap),
        "logits_mean": float(logits.mean()),
        "logits_std": float(logits.std()),
        "logits_min": float(logits.min()),
        "logits_max": float(logits.max())
    }
    
    if not eval_when_training:
        logger.info("Evaluation logits statistics:")
        logger.info(f"  Mean: {result['logits_mean']:.4f}, Std: {result['logits_std']:.4f}")
        logger.info(f"  Range: [{result['logits_min']:.4f}, {result['logits_max']:.4f}]")
        logger.info(f"  Predictions >0.5: {preds_05.sum()}/{len(preds_05)}")
    
    return result
def _test_debug(args, model, tokenizer):
    # Load test dataset
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    print(f"DEBUG: Test dataset length: {len(eval_dataset)}")
    print(f"DEBUG: Test file: {args.test_data_file}")
    print(f"DEBUG: Model type: {args.model_type}")
    print(f"DEBUG: Model class: {type(model)}")
    
    if len(eval_dataset) == 0:
        print("ERROR: Test dataset is empty!")
        return {"test_acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    eval_sampler = (
        SequentialSampler(eval_dataset)
        if args.local_rank == -1
        else DistributedSampler(eval_dataset)
    )
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    print(f"DEBUG: Dataloader length: {len(eval_dataloader)}")

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    logits = []
    labels = []
    failed_batches = 0
    successful_batches = 0
    
    for batch_idx, batch in enumerate(tqdm(eval_dataloader, total=len(eval_dataloader))):
        try:
            inputs = batch[0].to(args.device)
            label = batch[1].to(args.device)
            
            print(f"DEBUG Batch {batch_idx}: input shape {inputs.shape}, label shape {label.shape}")
            
            with torch.no_grad():
                # Test the forward pass
                try:
                    logit = model(inputs)
                    print(f"DEBUG Batch {batch_idx}: Forward pass successful")
                    print(f"DEBUG Batch {batch_idx}: Output type {type(logit)}")
                    
                    if isinstance(logit, tuple):
                        print(f"DEBUG Batch {batch_idx}: Tuple output, taking first element")
                        logit = logit[0]
                    
                    print(f"DEBUG Batch {batch_idx}: Final logit shape {logit.shape}")
                    
                    # Convert to numpy
                    logit_np = logit.cpu().numpy()
                    label_np = label.cpu().numpy()
                    
                    print(f"DEBUG Batch {batch_idx}: Numpy conversion successful")
                    print(f"DEBUG Batch {batch_idx}: Logit numpy shape {logit_np.shape}")
                    print(f"DEBUG Batch {batch_idx}: Label numpy shape {label_np.shape}")
                    
                    logits.append(logit_np)
                    labels.append(label_np)
                    successful_batches += 1
                    
                except Exception as forward_error:
                    print(f"ERROR Batch {batch_idx}: Forward pass failed: {forward_error}")
                    print(f"ERROR Batch {batch_idx}: Error type: {type(forward_error)}")
                    import traceback
                    traceback.print_exc()
                    failed_batches += 1
                    continue
                    
        except Exception as batch_error:
            print(f"ERROR Batch {batch_idx}: Batch processing failed: {batch_error}")
            import traceback
            traceback.print_exc()
            failed_batches += 1
            continue
        
        # Stop after a few batches for debugging
        if batch_idx >= 5:
            print(f"DEBUG: Stopping after {batch_idx + 1} batches for debugging")
            break

    print(f"DEBUG: Processing complete")
    print(f"DEBUG: Successful batches: {successful_batches}")
    print(f"DEBUG: Failed batches: {failed_batches}")
    print(f"DEBUG: Total logits collected: {len(logits)}")
    print(f"DEBUG: Total labels collected: {len(labels)}")

    if len(logits) == 0:
        print("ERROR: No logits collected - all batches failed")
        return {"test_acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Try concatenation with debug info
    
    # Print logits statistics for analysis
    logger.info("Logits statistics:")
    logger.info(f"  Shape: {logits.shape}")
    logger.info(f"  Min: {logits.min():.4f}, Max: {logits.max():.4f}")
    logger.info(f"  Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
    logger.info(f"  Label distribution: Positive={(labels==1).sum()}, Negative={(labels==0).sum()}")

def test(args, model, tokenizer, tb_writer=None):
    # Load test dataset
    eval_dataset = TextDataset(tokenizer, args, args.test_data_file)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    
    eval_sampler = (
        SequentialSampler(eval_dataset)
        if args.local_rank == -1
        else DistributedSampler(eval_dataset)
    )
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    model.eval()
    logits = []
    labels = []
    
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        inputs = batch[0].to(args.device)
        label = batch[1].to(args.device)
        with torch.no_grad():
            logit = model(inputs)
            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())

    logits = np.concatenate(logits, 0)
    labels = np.concatenate(labels, 0)
    
    # Print logits statistics for analysis
    logger.info("Logits statistics:")
    logger.info(f"  Shape: {logits.shape}")
    logger.info(f"  Min: {logits.min():.4f}, Max: {logits.max():.4f}")
    logger.info(f"  Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
    logger.info(f"  Label distribution: Positive={(labels==1).sum()}, Negative={(labels==0).sum()}")

    # Define threshold search space
    thresholds = np.arange(0.1, 0.9, 0.02)

    # Perform threshold optimization based on selected method
    logger.info("***** Threshold Optimization *****")
    logger.info(f"Method: {args.threshold_method}")
    logger.info(f"Optimization metric: {args.threshold_metric}")

    if args.threshold_method == "both":
        # Compare both methods
        comparison = compare_threshold_methods(
            logits, labels, thresholds,
            optimization_metric=args.threshold_metric,
            ghost_metric=args.threshold_metric if args.threshold_metric in ["kappa", "mcc", "f1"] else "kappa",
            n_subsets=args.ghost_n_subsets,
            random_seed=args.seed
        )

        # Use GHOST results as the "optimal" method when comparing both
        best_threshold = comparison["ghost"]["threshold"]
        best_metrics = comparison["ghost"]["metrics"]
        default_metrics = comparison["default_metrics"]

        # Store both for detailed output
        grid_threshold = comparison["grid_search"]["threshold"]
        grid_metrics = comparison["grid_search"]["metrics"]
        threshold_results = comparison["grid_search"]["all_results"]
        ghost_stats = comparison["ghost"]["statistics"]

    elif args.threshold_method == "ghost":
        # Use GHOST method only
        best_threshold, best_metrics, ghost_stats = ghost_threshold_optimization(
            logits, labels, thresholds,
            optimization_metric=args.threshold_metric if args.threshold_metric in ["kappa", "mcc", "f1"] else "kappa",
            n_subsets=args.ghost_n_subsets,
            subset_size=args.ghost_subset_size,
            random_seed=args.seed
        )
        default_metrics = calculate_metrics_with_threshold(0.5, logits, labels)
        threshold_results = None

    else:  # grid_search (default)
        best_threshold, best_metrics, threshold_results = grid_search_threshold(
            logits, labels, thresholds,
            optimization_metric=args.threshold_metric,
            min_recall=args.min_recall,
            precision_weight=args.threshold_precision_weight
        )
        default_metrics = calculate_metrics_with_threshold(0.5, logits, labels)

    # Log results
    logger.info("***** Threshold Analysis Results *****")
    logger.info(f"Method: {args.threshold_method}")
    logger.info(f"Optimization metric: {args.threshold_metric}")
    if args.threshold_metric == "precision":
        logger.info(f"Min recall constraint: {args.min_recall:.2f}, Threshold precision weight: {args.threshold_precision_weight:.1f}")
    logger.info(f"Default threshold (0.5): F1={default_metrics['f1']:.4f}, Precision={default_metrics['precision']:.4f}, Recall={default_metrics['recall']:.4f}")
    logger.info(f"Optimal threshold ({best_threshold:.3f}): F1={best_metrics['f1']:.4f}, Precision={best_metrics['precision']:.4f}, Recall={best_metrics['recall']:.4f}")
    
    # Save detailed threshold analysis
    if threshold_results is not None:
        threshold_file = os.path.join(args.output_dir, "threshold_analysis.txt")
        with open(threshold_file, "w") as f:
            f.write("Threshold\tAccuracy\tPrecision\tRecall\tF1\tMCC\tKappa\tTP\tFP\tFN\tTN\n")
            for metrics in threshold_results:
                f.write(f"{metrics['threshold']:.3f}\t{metrics['accuracy']:.4f}\t"
                       f"{metrics['precision']:.4f}\t{metrics['recall']:.4f}\t{metrics['f1']:.4f}\t"
                       f"{metrics['mcc']:.4f}\t{metrics['kappa']:.4f}\t"
                       f"{metrics['true_pos']}\t{metrics['false_pos']}\t"
                       f"{metrics['false_neg']}\t{metrics['true_neg']}\n")

    # Use optimal threshold for final predictions
    final_preds = logits[:, 0] > best_threshold

    # Save predictions
    with open(os.path.join(args.output_dir, "predictions.txt"), "w") as f:
        for example, pred in zip(eval_dataset.examples, final_preds):
            f.write(f"{example.idx}\t{int(pred)}\n")

    # Save comparison of thresholds
    comparison_file = os.path.join(args.output_dir, "threshold_comparison.txt")
    with open(comparison_file, "w") as f:
        f.write("=== THRESHOLD OPTIMIZATION REPORT ===\n")
        f.write(f"Method: {args.threshold_method}\n")
        f.write(f"Optimization metric: {args.threshold_metric}\n")
        if args.threshold_metric == "precision":
            f.write(f"Min recall constraint: {args.min_recall:.2f}\n")
            f.write(f"Threshold precision weight: {args.threshold_precision_weight:.1f}\n")

        # Add GHOST-specific parameters if applicable
        if args.threshold_method in ["ghost", "both"]:
            f.write(f"\nGHOST Parameters:\n")
            f.write(f"  Number of subsets: {args.ghost_n_subsets}\n")
            f.write(f"  Subset size: {args.ghost_subset_size}\n")
            if 'ghost_stats' in locals():
                f.write(f"  Median {args.threshold_metric}: {ghost_stats['optimal_median_score']:.4f}±{ghost_stats['optimal_std_score']:.4f}\n")

        f.write(f"\nData Statistics:\n")
        f.write(f"Logits range: [{logits.min():.4f}, {logits.max():.4f}]\n")
        f.write(f"Logits mean±std: {logits.mean():.4f}±{logits.std():.4f}\n")
        f.write(f"Label distribution: {(labels==1).sum()} positive, {(labels==0).sum()} negative\n\n")

        f.write("Default Threshold (0.5):\n")
        for key, value in default_metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")

        f.write(f"\nOptimal Threshold ({best_threshold:.3f}) [{args.threshold_method}]:\n")
        for key, value in best_metrics.items():
            if isinstance(value, float):
                f.write(f"  {key}: {value:.4f}\n")
            else:
                f.write(f"  {key}: {value}\n")

        # Add comparison for "both" method
        if args.threshold_method == "both" and 'grid_threshold' in locals():
            f.write(f"\nGrid Search Threshold ({grid_threshold:.3f}):\n")
            for key, value in grid_metrics.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")

        f.write(f"\nImprovement over default (0.5):\n")
        f.write(f"  F1: {best_metrics['f1'] - default_metrics['f1']:+.4f}\n")
        f.write(f"  Precision: {best_metrics['precision'] - default_metrics['precision']:+.4f}\n")
        f.write(f"  Recall: {best_metrics['recall'] - default_metrics['recall']:+.4f}\n")
        f.write(f"  MCC: {best_metrics['mcc'] - default_metrics['mcc']:+.4f}\n")
        f.write(f"  Kappa: {best_metrics['kappa'] - default_metrics['kappa']:+.4f}\n")

    # Save as JSON for easier parsing
    comparison_json = {
        "method": args.threshold_method,
        "optimization_metric": args.threshold_metric,
        "data_stats": {
            "logits_min": float(logits.min()),
            "logits_max": float(logits.max()),
            "logits_mean": float(logits.mean()),
            "logits_std": float(logits.std()),
            "n_positive": int((labels==1).sum()),
            "n_negative": int((labels==0).sum()),
            "n_total": int(len(labels))
        },
        "default_threshold": {
            "threshold": 0.5,
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                       for k, v in default_metrics.items()}
        },
        "optimal_threshold": {
            "threshold": float(best_threshold),
            "method_used": args.threshold_method,
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                       for k, v in best_metrics.items()}
        },
        "improvement": {
            "f1": float(best_metrics['f1'] - default_metrics['f1']),
            "precision": float(best_metrics['precision'] - default_metrics['precision']),
            "recall": float(best_metrics['recall'] - default_metrics['recall']),
            "accuracy": float(best_metrics['accuracy'] - default_metrics['accuracy']),
            "mcc": float(best_metrics['mcc'] - default_metrics['mcc']),
            "kappa": float(best_metrics['kappa'] - default_metrics['kappa'])
        }
    }

    # Add GHOST-specific info if applicable
    if args.threshold_method in ["ghost", "both"]:
        comparison_json["ghost_params"] = {
            "n_subsets": args.ghost_n_subsets,
            "subset_size": args.ghost_subset_size
        }
        if 'ghost_stats' in locals():
            comparison_json["ghost_stats"] = {
                "optimal_median_score": float(ghost_stats['optimal_median_score']),
                "optimal_std_score": float(ghost_stats['optimal_std_score'])
            }

    # Add grid search comparison if using "both"
    if args.threshold_method == "both" and 'grid_threshold' in locals():
        comparison_json["grid_search_threshold"] = {
            "threshold": float(grid_threshold),
            "metrics": {k: float(v) if isinstance(v, (float, np.floating)) else int(v)
                       for k, v in grid_metrics.items()}
        }

    # Add precision optimization params if applicable
    if args.threshold_metric == "precision":
        comparison_json["precision_optimization"] = {
            "min_recall": args.min_recall,
            "precision_weight": args.threshold_precision_weight
        }

    json_file = os.path.join(args.output_dir, "threshold_results.json")
    with open(json_file, "w") as f:
        json.dump(comparison_json, f, indent=2)

    # Return both results for comparison
    result = {
        "test_acc": round(best_metrics["accuracy"], 4),
        "precision": round(best_metrics["precision"], 4),
        "recall": round(best_metrics["recall"], 4),
        "f1": round(best_metrics["f1"], 4),
        "optimal_threshold": round(best_threshold, 4),
        "default_f1": round(default_metrics["f1"], 4),
        "improvement": round(best_metrics["f1"] - default_metrics["f1"], 4)
    }
    
    logger.info("***** Final Test Results *****")
    for key, value in result.items():
        logger.info(f"  {key} = {value}")

    if args.use_wandb and args.local_rank in [-1, 0]:
        wandb.log({
            "test/accuracy": result["test_acc"],
            "test/precision": result["precision"],
            "test/recall": result["recall"],
            "test/f1": result["f1"],
            "test/optimal_threshold": result["optimal_threshold"],
            "test/improvement": result["improvement"]
        })

    # Log test results to tensorboard
    if tb_writer is not None and args.local_rank in [-1, 0]:
        tb_writer.add_scalar("test/accuracy", result["test_acc"], 0)
        tb_writer.add_scalar("test/precision", result["precision"], 0)
        tb_writer.add_scalar("test/recall", result["recall"], 0)
        tb_writer.add_scalar("test/f1", result["f1"], 0)
        tb_writer.add_scalar("test/optimal_threshold", result["optimal_threshold"], 0)
        tb_writer.add_scalar("test/improvement", result["improvement"], 0)
        
        # Log threshold analysis table
        #threshold_results = []  
        wandb.log({
            "test/threshold_analysis": wandb.Table(
                columns=["threshold", "accuracy", "precision", "recall", "f1"],
                data=[[t["threshold"], t["accuracy"], t["precision"], t["recall"], t["f1"]] 
                      for t in threshold_results[:20]]  # Log top 20 thresholds
            )
        })
    
    return result

def copy_split_metadata_to_output(args):
    """
    Copy data split metadata to the output directory for this experiment
    """
    if not args.one_data_file:
        return  # Only relevant when using auto-splitting
    
    # Find the metadata file
    input_dir = os.path.dirname(args.one_data_file)
    base_name = os.path.basename(args.one_data_file).replace('.jsonl', '')
    source_metadata = os.path.join(input_dir, f"{base_name}_split_metadata_seed{args.seed}.json")
    
    if not os.path.exists(source_metadata):
        print(f"Warning: Could not find split metadata at {source_metadata}")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Copy metadata to output directory
    dest_metadata = os.path.join(args.output_dir, "data_split_info.json")
    
    # Load, enhance, and save the metadata
    with open(source_metadata, 'r') as f:
        metadata = json.load(f)
    
    # Add experiment-specific information
    metadata.update({
        "experiment_output_dir": args.output_dir,
        "model_type": args.model_type,
        "model_name": args.model_name_or_path,
        "copied_at": datetime.now().isoformat(),
        "training_args": {
            "learning_rate": args.learning_rate,
            "batch_size": args.train_batch_size,
            "epochs": args.epoch,
            "pos_weight": args.pos_weight,
            "dropout_probability": args.dropout_probability,
            "block_size": args.block_size,
        }
    })
    
    with open(dest_metadata, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Copied split metadata to: {dest_metadata}")
    
    # Also create a simple summary file
    summary_file = os.path.join(args.output_dir, "experiment_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Experiment Summary\n")
        f.write(f"==================\n\n")
        f.write(f"Dataset: {base_name}\n")
        f.write(f"Model: {args.model_type} ({args.model_name_or_path})\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Split: {metadata['sizes']['train']}/{metadata['sizes']['val']}/{metadata['sizes']['test']} (train/val/test)\n")
        f.write(f"Total examples: {metadata['total_examples']}\n")
        f.write(f"Ratios: {metadata['ratios']['train']:.1%}/{metadata['ratios']['val']:.1%}/{metadata['ratios']['test']:.1%}\n\n")
        f.write(f"Training Parameters:\n")
        f.write(f"  Learning rate: {args.learning_rate}\n")
        f.write(f"  Batch size: {args.train_batch_size}\n")
        f.write(f"  Epochs: {args.epoch}\n")
        f.write(f"  Pos weight: {args.pos_weight}\n")
        f.write(f"  Dropout: {args.dropout_probability}\n")
        f.write(f"  Block size: {args.block_size}\n\n")
        f.write(f"Data Files:\n")
        f.write(f"  Original: {metadata['original_file']}\n")
        f.write(f"  Train: {args.train_data_file}\n")
        f.write(f"  Val: {args.eval_data_file}\n")
        f.write(f"  Test: {args.test_data_file}\n")
    
    print(f"Created experiment summary: {summary_file}")

def split_data_by_seed(input_file, seed, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split a single data file into train/val/test based on seed
    Saves splits in the same directory as the input file
    
    Args:
        input_file: Path to the input .jsonl file
        seed: Random seed for reproducible splits
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)  
        test_ratio: Proportion for testing (default 0.1)
    
    Returns:
        tuple: (train_file, val_file, test_file) paths
    """
    # Use the same directory as input file
    input_dir = os.path.dirname(input_file)
    base_name = os.path.basename(input_file).replace('.jsonl', '')
    
    # Check if splits already exist
    train_file = os.path.join(input_dir, f"{base_name}_train_seed{seed}.jsonl")
    val_file = os.path.join(input_dir, f"{base_name}_val_seed{seed}.jsonl")
    test_file = os.path.join(input_dir, f"{base_name}_test_seed{seed}.jsonl")
    
    # If splits already exist, just return the paths
    if all(os.path.exists(f) for f in [train_file, val_file, test_file]):
        print(f"Using existing splits for seed {seed}:")
        print(f"  Train: {train_file}")
        print(f"  Val: {val_file}")
        print(f"  Test: {test_file}")
        return train_file, val_file, test_file
    
    print(f"Creating new splits for seed {seed}...")
    
    # Load and split data
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"Loaded {len(data)} examples from {input_file}")
    
    # Set seed for reproducible splits
    random.seed(seed)
    
    # First split: train vs (val + test)
    train_data, temp_data = train_test_split(
        data, 
        train_size=train_ratio, 
        random_state=seed,
        stratify=[item['target'] for item in data] if 'target' in data[0] else None
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_data, test_data = train_test_split(
        temp_data,
        train_size=val_size,
        random_state=seed,
        stratify=[item['target'] for item in temp_data] if 'target' in temp_data[0] else None
    )
    
    # Save splits
    def save_jsonl(data, filename):
        with open(filename, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    
    save_jsonl(train_data, train_file)
    save_jsonl(val_data, val_file)
    save_jsonl(test_data, test_file)
    
    print(f"Split complete:")
    print(f"  Train: {len(train_data)} examples -> {train_file}")
    print(f"  Val:   {len(val_data)} examples -> {val_file}")
    print(f"  Test:  {len(test_data)} examples -> {test_file}")
    
    # Save split metadata in data directory (for reuse)
    metadata = {
        "original_file": input_file,
        "seed": seed,
        "ratios": {"train": train_ratio, "val": val_ratio, "test": test_ratio},
        "sizes": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
        "total_examples": len(data),
        "split_files": {
            "train": train_file,
            "val": val_file, 
            "test": test_file
        },
        "created_at": datetime.now().isoformat()
    }
    
    metadata_file = os.path.join(input_dir, f"{base_name}_split_metadata_seed{seed}.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata: {metadata_file}")
    
    return train_file, val_file, test_file


def main():
    parser = argparse.ArgumentParser()
    data_group = parser.add_mutually_exclusive_group(required=True)

    ## Required parameters
    data_group.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        help="The input training data file (a text file).",
    )

    data_group.add_argument(
        "--one_data_file",
        default=None,
        type=str,
        help="Automate splitting of data based on seed",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters

    parser.add_argument(
        "--max_source_length", 
        default=400, 
        type=int, 
        help="Maximum source sequence length for NatGen model"
    )
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )

    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        help="The model architecture to be fine-tuned. bert | codet5",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization.",
    )

    parser.add_argument(
        "--mlm",
        action="store_true",
    help="Train with masked-language modeling loss instead of language modeling.",
)
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )

    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=1.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=50, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--epoch", type=int, default=42, help="random seed for initialization"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )

    # Add early stopping parameters and dropout probability parameters
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=None,
        help="Number of epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument(
        "--min_loss_delta",
        type=float,
        default=0.001,
        help="Minimum change in the loss required to qualify as an improvement.",
    )
    parser.add_argument(
        "--dropout_probability", type=float, default=0, help="dropout probability"
    )
    parser.add_argument(
        "--pos_weight",
        default=1.0,
        type=float,
        help="Weight for positive class in loss function during TRAINING. "
             "Higher values (>1.0) prioritize recall, lower values (<1.0) prioritize precision. "
             "Example: pos_weight=0.4 trains model to reduce false positives."
    )

    # Loss function selection
    parser.add_argument(
        "--loss_type",
        default="bce",
        type=str,
        choices=["bce", "cb_focal"],
        help="Loss function to use: 'bce' (standard binary cross-entropy with pos_weight) "
             "or 'cb_focal' (Class-Balanced Focal Loss). Default: 'bce'"
    )
    parser.add_argument(
        "--cb_beta",
        default=0.9999,
        type=float,
        help="Class-balanced loss beta parameter (only used with loss_type='cb_focal'). "
             "Controls the effective number of samples. Default: 0.9999"
    )
    parser.add_argument(
        "--focal_gamma",
        default=2.0,
        type=float,
        help="Focal loss gamma parameter (only used with loss_type='cb_focal'). "
             "Controls focusing on hard examples. Default: 2.0"
    )

    # Threshold optimization args (POST-TRAINING inference)
    parser.add_argument(
        "--threshold_method",
        default="grid_search",
        type=str,
        choices=["grid_search", "ghost", "both"],
        help="Threshold optimization method: "
             "'grid_search' (default, simple), 'ghost' (GHOST: Generalized tHreshOld ShifTing), "
             "or 'both' (compare both methods). "
             "GHOST uses bootstrap aggregation for more robust threshold selection."
    )
    parser.add_argument(
        "--threshold_metric",
        default="f1",
        type=str,
        choices=["f1", "precision", "kappa", "mcc"],
        help="Metric to optimize when selecting threshold at INFERENCE time: "
             "'f1' (default, balanced), 'precision' (reduce false positives), "
             "'kappa' (Cohen's Kappa), or 'mcc' (Matthews Correlation Coefficient). "
             "This is independent of pos_weight used during training."
    )
    parser.add_argument(
        "--min_recall",
        default=0.5,
        type=float,
        help="Minimum recall constraint when threshold_metric='precision'. "
             "Ensures we still catch at least this fraction of vulnerabilities. "
             "Default: 0.5 (catch ≥50%% of bugs)"
    )
    parser.add_argument(
        "--threshold_precision_weight",
        default=2.0,
        type=float,
        help="Weight for precision in threshold scoring when threshold_metric='precision'. "
             "Higher values more aggressively favor precision. Default: 2.0. "
             "NOTE: This is different from pos_weight (which affects training)."
    )
    parser.add_argument(
        "--ghost_n_subsets",
        default=100,
        type=int,
        help="Number of bootstrap subsets for GHOST threshold optimization. Default: 100"
    )
    parser.add_argument(
        "--ghost_subset_size",
        default=0.8,
        type=float,
        help="Proportion of samples per subset for GHOST (0-1). Default: 0.8"
    )

    #wandb args
    parser.add_argument(
        "--wandb_project", 
        default="vulnerability-detection-benchmark", 
        type=str, 
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_run_name", 
        default=None, 
        type=str, 
        help="Wandb run name (auto-generated if None)"
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use wandb logging"
    )
    parser.add_argument(
        "--use_tensorboard",
        action="store_true",
        help="Whether to use tensorboard logging"
    )
    parser.add_argument(
        "--allow_pretrained_only",
        action="store_true",
        help="Allow testing with pretrained model only (no fine-tuned checkpoint required)"
    )
    parser.add_argument(
        "--tensorboard_log_dir",
        default="logs",
        type=str,
        help="TensorBoard log directory"
    )

    args = parser.parse_args()
    print(args)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    args.per_gpu_train_batch_size = args.train_batch_size // args.n_gpu
    args.per_gpu_eval_batch_size = args.eval_batch_size // args.n_gpu
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, "checkpoint-last")
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, "pytorch_model.bin")
        args.config_name = os.path.join(checkpoint_last, "config.json")
        idx_file = os.path.join(checkpoint_last, "idx_file.txt")
        with open(idx_file, encoding="utf-8") as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, "step_file.txt")
        if os.path.exists(step_file):
            with open(step_file, encoding="utf-8") as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info(
            "reload model from {}, resume from {} epoch".format(
                checkpoint_last, args.start_epoch
            )
        )

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # Set num_labels based on model type
    # LineVul and natgen use 2-class classifier, others use single output with sigmoid
    if args.model_type in ["linevul", "natgen"]:
        config.num_labels = 2
    else:
        config.num_labels = 1
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    if args.block_size <= 0:
        args.block_size = (
            tokenizer.max_len_single_sentence
        )  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
    else:
        model = model_class(config)

    if args.model_type == "codet5":
        # Use CodeT5 model
        from model import CodeT5Model
        model = CodeT5Model(model, config, tokenizer, args)
    elif args.model_type == "codet5_full":
        # Use CodeT5 model
        from model import CodeT5FullModel
        model = CodeT5FullModel(model, config, tokenizer, args)
    elif args.model_type == "natgen":
        # Use Natgen model
        from model import DefectModel
        model = DefectModel(model, config, tokenizer, args)
    elif args.model_type == "linevul":
        # Use LineVul model with 2-class classifier
        from model import LineVulModel
        model = LineVulModel(model, config, tokenizer, args)
    else:
        model = Model(model, config, tokenizer, args)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    if args.one_data_file:
        print(f"Splitting data from: {args.one_data_file}")
        train_file, val_file, test_file = split_data_by_seed(args.one_data_file, args.seed)
        
        args.train_data_file = train_file
        args.eval_data_file = val_file  
        args.test_data_file = test_file

        copy_split_metadata_to_output(args)

    logger.info("Training/evaluation parameters %s", args)

    if args.use_wandb and args.local_rank in [-1, 0]:
        # Auto-generate run name if not provided
        if args.wandb_run_name is None:
            args.wandb_run_name = f"{args.model_type}_{args.train_data_file.split('/')[-2]}_{args.pos_weight}_seed{args.seed}"
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_type": args.model_type,
                "model_name": args.model_name_or_path,
                "dataset": args.train_data_file.split('/')[-2] if args.train_data_file else "unknown",
                "learning_rate": args.learning_rate,
                "batch_size": args.train_batch_size,
                "epochs": args.epoch,
                "pos_weight": args.pos_weight,
                "dropout_probability": args.dropout_probability,
                "block_size": args.block_size,
                "seed": args.seed,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "max_grad_norm": args.max_grad_norm,
                "weight_decay": args.weight_decay,
                "warmup_steps": args.warmup_steps,
            },
            tags=[args.model_type, args.train_data_file.split('/')[-2] if args.train_data_file else "unknown"]
        )
        
        wandb.watch(model, log="all", log_freq=100)

    # Initialize TensorBoard writer if requested
    tb_writer = None
    if args.use_tensorboard and args.local_rank in [-1, 0]:
        # Create run-specific log directory
        tb_run_name = f"{args.model_type}_{args.train_data_file.split('/')[-2] if args.train_data_file else 'unknown'}_{args.pos_weight}_seed{args.seed}"
        tb_log_dir = os.path.join(args.tensorboard_log_dir, tb_run_name)
        tb_writer = SummaryWriter(log_dir=tb_log_dir)

        # Log hyperparameters
        hparams = {
            "model_type": args.model_type,
            "learning_rate": args.learning_rate,
            "batch_size": args.train_batch_size,
            "epochs": args.epoch,
            "pos_weight": args.pos_weight,
            "dropout_probability": args.dropout_probability,
            "block_size": args.block_size,
            "seed": args.seed,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_grad_norm": args.max_grad_norm,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
        }
        tb_writer.add_hparams(hparams, {})

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = TextDataset(tokenizer, args, args.train_data_file)
        if args.local_rank == 0:
            torch.distributed.barrier()

        # Compute class distribution for CB-Focal loss
        if args.loss_type == 'cb_focal':
            labels = [example.label for example in train_dataset.examples]
            num_negative = sum(1 for label in labels if label == 0)
            num_positive = sum(1 for label in labels if label == 1)
            args.samples_per_cls = torch.tensor([num_negative, num_positive], dtype=torch.float32)
            logger.info(f"Class distribution for CB-Focal loss: neg={num_negative}, pos={num_positive}")

            # Update model with class distribution
            model.samples_per_cls = args.samples_per_cls
        else:
            args.samples_per_cls = None

        train(args, train_dataset, model, tokenizer, tb_writer)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = "checkpoint-best-acc/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))

        # Load checkpoint or validate pretrained-only usage
        if os.path.exists(output_dir):
            logger.info(f"Loading checkpoint from {output_dir}")
            model.load_state_dict(torch.load(output_dir))
            model.to(args.device)
        else:
            # Checkpoint not found - determine if this is expected
            if args.do_train:
                raise RuntimeError(
                    f"Expected checkpoint at {output_dir} but not found. "
                    f"Training was requested but checkpoint is missing."
                )
            elif not args.allow_pretrained_only:
                raise RuntimeError(
                    f"No checkpoint found at {output_dir}. "
                    f"To test with pretrained model only, use --allow_pretrained_only flag."
                )
            else:
                logger.info(f"No checkpoint found at {output_dir}, using pretrained model from {args.model_name_or_path} (--allow_pretrained_only specified)")
                model.to(args.device)

        result = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

        # Log final evaluation results to tensorboard
        if tb_writer is not None:
            for key, value in result.items():
                tb_writer.add_scalar(f"final_eval/{key}", value, 0)

    if args.do_test and args.local_rank in [-1, 0]:
        checkpoint_prefix = "checkpoint-best-acc/model.bin"
        output_dir = os.path.join(args.output_dir, "{}".format(checkpoint_prefix))

        # Load checkpoint or validate pretrained-only usage
        if os.path.exists(output_dir):
            logger.info(f"Loading checkpoint from {output_dir}")
            model.load_state_dict(torch.load(output_dir))
            model.to(args.device)
        else:
            # Checkpoint not found - determine if this is expected
            if args.do_train:
                raise RuntimeError(
                    f"Expected checkpoint at {output_dir} but not found. "
                    f"Training was requested but checkpoint is missing."
                )
            elif not args.allow_pretrained_only:
                raise RuntimeError(
                    f"No checkpoint found at {output_dir}. "
                    f"To test with pretrained model only, use --allow_pretrained_only flag."
                )
            else:
                logger.info(f"No checkpoint found at {output_dir}, using pretrained model from {args.model_name_or_path} (--allow_pretrained_only specified)")
                model.to(args.device)

        test(args, model, tokenizer, tb_writer)

    if args.use_wandb and args.local_rank in [-1, 0]:
        wandb.finish()

    if tb_writer is not None:
        tb_writer.close()

    return results


if __name__ == "__main__":
    from error_logger import error_logging_context, get_batch_id_from_env

    # Get batch ID from environment (set by runner.py or use timestamp)
    batch_id = get_batch_id_from_env()

    # Run main with error logging
    with error_logging_context(batch_id=batch_id):
        main()
