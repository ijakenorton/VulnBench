# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from torch.nn import CrossEntropyLoss, MSELoss


def cb_focal_loss(logits, labels, samples_per_cls, beta=0.9999, gamma=2.0):
    """
    Class-Balanced Focal Loss

    Args:
        logits: Raw logits from model (before sigmoid)
        labels: Binary labels (0 or 1)
        samples_per_cls: List/tensor with [num_negative_samples, num_positive_samples]
        beta: Class-balanced loss hyperparameter (default: 0.9999)
        gamma: Focal loss focusing parameter (default: 2.0)

    Returns:
        loss: Scalar loss value
    """
    labels = labels.float()

    # Ensure samples_per_cls is on the same device as labels
    if not isinstance(samples_per_cls, torch.Tensor):
        samples_per_cls = torch.tensor(samples_per_cls, dtype=torch.float32)
    samples_per_cls = samples_per_cls.to(labels.device)

    # Calculate effective number of samples
    effective_num = 1.0 - torch.pow(beta, samples_per_cls)
    weights = (1.0 - beta) / effective_num
    weights = weights / weights.sum() * 2.0  # Normalize to sum to 2 (for binary case)

    # Get class weights for each sample
    # weights[0] for negative class (label=0), weights[1] for positive class (label=1)
    sample_weights = labels * weights[1] + (1 - labels) * weights[0]

    # Calculate focal loss components
    probs = torch.sigmoid(logits)

    # Binary cross-entropy
    bce = -(labels * torch.log(probs + 1e-10) + (1 - labels) * torch.log(1 - probs + 1e-10))

    # Focal modulation factor
    # pt is the probability of the true class
    pt = labels * probs + (1 - labels) * (1 - probs)
    focal_weight = torch.pow(1 - pt, gamma)

    # Combine all components
    loss = focal_weight * bce * sample_weights

    return loss.mean()


def compute_loss(logits, labels, args, samples_per_cls=None):
    """
    Unified loss computation function that supports multiple loss types.

    Args:
        logits: Raw logits from model (before sigmoid)
        labels: Binary labels (0 or 1)
        args: Arguments object containing loss configuration
        samples_per_cls: For CB-Focal loss, list/tensor with [num_neg, num_pos]

    Returns:
        loss: Scalar loss value
    """
    loss_type = getattr(args, 'loss_type', 'bce')

    if loss_type == 'cb_focal':
        # CB-Focal Loss
        if samples_per_cls is None:
            raise ValueError("samples_per_cls must be provided for cb_focal loss")
        beta = getattr(args, 'cb_beta', 0.9999)
        gamma = getattr(args, 'focal_gamma', 2.0)
        return cb_focal_loss(logits, labels, samples_per_cls, beta=beta, gamma=gamma)
    else:
        # Standard BCE with pos_weight (default)
        pos_weight = torch.tensor(getattr(args, 'pos_weight', 1.0)).to(labels.device)
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=pos_weight
        )



class Model(nn.Module):
    def __init__(self, encoder,config,tokenizer,args):
        super(Model, self).__init__()
        self.encoder = encoder
        self.config=config
        self.tokenizer=tokenizer
        self.args=args

        # Define dropout layer, dropout_probability is taken from args.
        self.dropout = nn.Dropout(args.dropout_probability)

        # Store class distribution for CB-Focal loss
        self.samples_per_cls = getattr(args, 'samples_per_cls', None)

        
    def _forward(self, input_ids=None,labels=None): 
        outputs=self.encoder(input_ids,attention_mask=input_ids.ne(1))[0]

        # Apply dropout
        outputs = self.dropout(outputs)

        logits=outputs
        prob=torch.sigmoid(logits)
        if labels is not None:
            labels=labels.float()
            loss=torch.log(prob[:,0]+1e-10)*labels+torch.log((1-prob)[:,0]+1e-10)*(1-labels)
            loss=-loss.mean()
            return loss,prob
        else:
            return prob

    def forward(self, input_ids=None, labels=None):
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))[0]
        outputs = self.dropout(outputs)

        logits = outputs  # Keep as raw logits
        prob = torch.sigmoid(logits)

        if labels is not None:
            labels = labels.float()

            # Use unified loss computation
            loss = compute_loss(logits[:, 0], labels, self.args, self.samples_per_cls)

            return loss, prob
        else:
            return prob


class LineVulModel(nn.Module):
    """Model wrapper for LineVul with 2-class classifier head"""
    def __init__(self, encoder, config, tokenizer, args):
        super(LineVulModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        # Define dropout layer, dropout_probability is taken from args
        self.dropout = nn.Dropout(args.dropout_probability)

        # Store class distribution for CB-Focal loss
        self.samples_per_cls = getattr(args, 'samples_per_cls', None)

    def forward(self, input_ids=None, labels=None):
        # Get logits from encoder (shape: [batch_size, 2])
        outputs = self.encoder(input_ids, attention_mask=input_ids.ne(1))
        logits = outputs.logits

        # Apply dropout
        logits = self.dropout(logits)

        # Get probabilities using softmax for 2-class classification
        probs = torch.softmax(logits, dim=-1)

        # Extract vulnerability probability (class 1)
        prob = probs[:, 1:2]  # Keep dimension for consistency with other models

        if labels is not None:
            labels = labels.float()

            # Use unified loss computation
            loss = compute_loss(logits[:, 1], labels, self.args, self.samples_per_cls)

            return loss, prob
        else:
            return prob

from transformers import T5EncoderModel, T5Tokenizer

class CodeT5Model(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5Model, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.dropout = nn.Dropout(getattr(args, 'dropout_probability', 0.1))

        # Respect config.num_labels - default to 1 for backwards compatibility
        num_labels = getattr(config, 'num_labels', 1)
        # CodeT5-base has 768 hidden size, same as CodeBERT
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # Store class distribution for CB-Focal loss
        self.samples_per_cls = getattr(args, 'samples_per_cls', None)
        
    def forward(self, input_ids=None, labels=None):
        # Use only the encoder part of CodeT5
        encoder_outputs = self.encoder( input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id)) 
        outputs = encoder_outputs.last_hidden_state
        
        # Apply dropout
        outputs = self.dropout(outputs)
        
        # Pool the sequence (mean pooling over sequence length)
        # Mask padding tokens
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).float()
        masked_outputs = outputs * attention_mask.unsqueeze(-1)
        summed = torch.sum(masked_outputs, dim=1)
        lengths = torch.sum(attention_mask, dim=1, keepdim=True)
        pooled_output = summed / lengths
        
        # Classification
        logits = self.classifier(pooled_output)
        prob = torch.sigmoid(logits)
        
        if labels is not None:
            labels = labels.float()

            # Use unified loss computation
            loss = compute_loss(logits[:, 0], labels, self.args, self.samples_per_cls)

            return loss, prob
        else:
            return prob

class CodeT5FullModel(nn.Module):
    """CodeT5 with full encoder-decoder - following NatGen pattern"""
    def __init__(self, encoder, config, tokenizer, args):
        super(CodeT5FullModel, self).__init__()
        self.encoder = encoder  # T5ForConditionalGeneration
        self.config = config
        self.tokenizer = tokenizer
        self.args = args

        self.dropout = nn.Dropout(getattr(args, 'dropout_probability', 0.1))
        # Respect config.num_labels - default to 2 for backwards compatibility
        num_labels = getattr(config, 'num_labels', 2)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        # Store class distribution for CB-Focal loss
        self.samples_per_cls = getattr(args, 'samples_per_cls', None)
        
    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)

        # Use the full T5 model with decoder - SAME as NatGen
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask,
                               output_hidden_states=True)

        # Now we can access decoder_hidden_states - SAME as NatGen
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        # Handle varying EOS tokens robustly - SAME as NatGen
        if eos_mask.sum() == 0:
            # No EOS tokens - use last non-padding token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = source_ids.size(0)
            batch_indices = torch.arange(batch_size, device=source_ids.device)
            vec = hidden_states[batch_indices, seq_lengths, :]
        else:
            # Handle varying EOS tokens per sequence
            batch_size, seq_len = source_ids.shape
            vec_list = []

            for i in range(batch_size):
                eos_positions = torch.where(eos_mask[i])[0]

                if len(eos_positions) > 0:
                    last_eos_pos = eos_positions[-1]
                    vec_list.append(hidden_states[i, last_eos_pos, :])
                else:
                    seq_length = attention_mask[i].sum() - 1
                    vec_list.append(hidden_states[i, seq_length, :])

            vec = torch.stack(vec_list)

        return vec
        
    def forward(self, input_ids=None, labels=None):
        # Reshape input like NatGen does
        input_ids = input_ids.view(-1, self.args.max_source_length)
        
        # Get T5 representation - SAME method for training and inference
        vec = self.get_t5_vec(input_ids)
        vec = self.dropout(vec)
        
        # Classify - same as NatGen
        logits = self.classifier(vec)
        
        if labels is not None:
            # Training mode - EXACT same as NatGen
            labels = labels.float()

            # Extract vulnerability logit (class 1 = vulnerable) - SAME as NatGen
            if logits.shape[1] == 2:
                binary_logits = logits[:, 1]
            else:
                binary_logits = logits.view(-1)

            # Use unified loss computation
            loss = compute_loss(binary_logits, labels, self.args, self.samples_per_cls)

            # Return probabilities in same format as other models - SAME as NatGen
            prob = torch.sigmoid(binary_logits.unsqueeze(1))
            return loss, prob
        else:
            # For inference - EXACT same as NatGen
            if logits.shape[1] == 2:
                binary_logits = logits[:, 1]
            else:
                binary_logits = logits.view(-1)
            prob = torch.sigmoid(binary_logits.unsqueeze(1))
            return prob

class DefectModel(nn.Module):
    def __init__(self, encoder, config, tokenizer, args):
        super(DefectModel, self).__init__()
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        # Respect config.num_labels - default to 2 for backwards compatibility
        num_labels = getattr(config, 'num_labels', 2)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.args = args

        # Store class distribution for CB-Focal loss
        self.samples_per_cls = getattr(args, 'samples_per_cls', None)

    def get_t5_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        
        # Use the full T5 model with decoder
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, 
                               output_hidden_states=True)
        
        # Now we can access decoder_hidden_states
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        # Handle varying EOS tokens robustly
        if eos_mask.sum() == 0:
            # No EOS tokens - use last non-padding token
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_size = source_ids.size(0)
            batch_indices = torch.arange(batch_size, device=source_ids.device)
            vec = hidden_states[batch_indices, seq_lengths, :]
        else:
            # Handle varying EOS tokens per sequence
            batch_size, seq_len = source_ids.shape
            vec_list = []
            
            for i in range(batch_size):
                eos_positions = torch.where(eos_mask[i])[0]
                
                if len(eos_positions) > 0:
                    last_eos_pos = eos_positions[-1]
                    vec_list.append(hidden_states[i, last_eos_pos, :])
                else:
                    seq_length = attention_mask[i].sum() - 1
                    vec_list.append(hidden_states[i, seq_length, :])
            
            vec = torch.stack(vec_list)
        
        return vec

    def get_bart_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        outputs = self.encoder(input_ids=source_ids, attention_mask=attention_mask,
                               labels=source_ids, decoder_attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs['decoder_hidden_states'][-1]
        eos_mask = source_ids.eq(self.config.eos_token_id)

        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        vec = hidden_states[eos_mask, :].view(hidden_states.size(0), -1,
                                              hidden_states.size(-1))[:, -1, :]
        return vec

    def get_roberta_vec(self, source_ids):
        attention_mask = source_ids.ne(self.tokenizer.pad_token_id)
        vec = self.encoder(input_ids=source_ids, attention_mask=attention_mask)[0][:, 0, :]
        return vec

    def forward(self, source_ids=None, labels=None):
        source_ids = source_ids.view(-1, self.args.max_source_length)

        if self.args.model_type == 'codet5' or self.args.model_type == 'natgen':
            vec = self.get_t5_vec(source_ids)
        elif self.args.model_type == 'bart':
            vec = self.get_bart_vec(source_ids)
        elif self.args.model_type == 'roberta':
            vec = self.get_roberta_vec(source_ids)

        logits = self.classifier(vec)
        
        if labels is not None:
            labels = labels.float()

            # Extract vulnerability logit (class 1 = vulnerable)
            if logits.shape[1] == 2:
                binary_logits = logits[:, 1]
            else:
                binary_logits = logits.view(-1)

            # Use unified loss computation
            loss = compute_loss(binary_logits, labels, self.args, self.samples_per_cls)

            # Return probabilities in same format as other models
            prob = torch.sigmoid(binary_logits.unsqueeze(1))
            return loss, prob
        else:
            # For inference
            if logits.shape[1] == 2:
                binary_logits = logits[:, 1]
            else:
                binary_logits = logits.view(-1)
            prob = torch.sigmoid(binary_logits.unsqueeze(1))
            return prob