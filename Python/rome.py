import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import cvxpy as cp

class MoE(nn.Module):
    def __init__(self, input_size_A, input_size_S, output_size, num_experts, expert_hidden_size, gating_hidden_size):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        
        self.experts = nn.ModuleList([
            MLP(input_size_A + input_size_S, output_size, expert_hidden_size)
            for _ in range(num_experts)
        ])
        
        self.gating_net = MLP(input_size_S, num_experts, gating_hidden_size, is_gating_net=True)
        
    def forward(self, A_data, S_data):
        gating_logits = self.gating_net(S_data)
        gating_probs = F.softmax(gating_logits, dim=1)
        
        combined_input = torch.cat([A_data, S_data], dim=-1)
        expert_outputs = torch.stack([expert(combined_input) for expert in self.experts], dim=-1) 
        
        if expert_outputs.shape[1] == 1:
            expert_outputs = expert_outputs.squeeze(1) 
        
        final_prediction = torch.sum(expert_outputs * gating_probs, dim=1, keepdim=True)
        
        return final_prediction

class FairMoE_S(nn.Module):
    def __init__(self, input_size_A, input_size_S, output_size, num_experts, expert_hidden_size, gating_hidden_size):

        super(FairMoE_S, self).__init__()
        self.num_experts = num_experts
        
        self.experts = nn.ModuleList([
            MLP(input_size_A, output_size, expert_hidden_size)
            for _ in range(num_experts)
        ])
        
        self.gating_net = MLP(input_size_S, num_experts, gating_hidden_size, is_gating_net=True)
        
    def forward(self, A_data, S_data):
        gating_logits = self.gating_net(S_data)
        gating_probs = F.softmax(gating_logits, dim=1) 
        
        expert_outputs = torch.stack([expert(A_data) for expert in self.experts], dim=-1) 
        
        if expert_outputs.shape[1] == 1: 
            expert_outputs = expert_outputs.squeeze(1) 
        
        final_prediction = torch.sum(expert_outputs * gating_probs, dim=1, keepdim=True)
        
        return final_prediction

class FairMoE_AS(nn.Module):
    def __init__(self, input_size_A, input_size_S, output_size, num_experts, expert_hidden_size, gating_hidden_size):
        super(FairMoE_AS, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            MLP(input_size_A, output_size, expert_hidden_size)
            for _ in range(num_experts)
        ])
        
        self.gating_net = MLP(input_size_A + input_size_S, num_experts, gating_hidden_size, is_gating_net=True)
        
    def forward(self, A_data, S_data):
        combined_input = torch.cat([A_data, S_data], dim=-1)
        gating_logits = self.gating_net(combined_input)
        gating_probs = F.softmax(gating_logits, dim=1)
    
        expert_outputs = torch.stack([expert(A_data) for expert in self.experts], dim=-1) 
        
        if expert_outputs.shape[1] == 1:
            expert_outputs = expert_outputs.squeeze(1) 
        
        final_prediction = torch.sum(expert_outputs * gating_probs, dim=1, keepdim=True)
        
        return final_prediction

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, is_gating_net=False):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.is_gating_net = is_gating_net 
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class BaselineMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(BaselineMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train_moe_model(batch_A, batch_S, batch_Y, model, loss_fn, optimizer, device):
    model.train()
    batch_A, batch_S, batch_Y = batch_A.to(device), batch_S.to(device), batch_Y.to(device)
    
    optimizer.zero_grad()
    predictions = model(batch_A, batch_S)
    loss = loss_fn(predictions, batch_Y)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_rome_moe_with_dro(batch_A, batch_S, batch_Y, model, optimizer, device, alpha=0.5):
    model.train()
    batch_A, batch_S, batch_Y = batch_A.to(device), batch_S.to(device), batch_Y.to(device)
    
    optimizer.zero_grad()
    predictions = model(batch_A, batch_S)

    with torch.no_grad():
        if isinstance(model, FairMoE_S):
            gating_logits = model.gating_net(batch_S)
        elif isinstance(model, FairMoE_AS):
            combined_input = torch.cat([batch_A, batch_S], dim=-1)
            gating_logits = model.gating_net(combined_input)
        else: 
            gating_logits = model.gating_net(batch_S)
        
        gating_probs = F.softmax(gating_logits, dim=1) 

    expert_losses = []
    for k in range(model.num_experts):
        expert_weights = gating_probs[:, k]
        significant_mask = expert_weights > 0.1 
        
        if significant_mask.sum() > 0:
            expert_mse = (expert_weights[significant_mask] * 
                         (predictions[significant_mask] - batch_Y[significant_mask])**2).mean()
            expert_losses.append(expert_mse)
    
    if len(expert_losses) > 0:
        avg_loss = F.mse_loss(predictions, batch_Y)
        worst_expert_loss = torch.max(torch.stack(expert_losses))
        
        loss = (1 - alpha) * avg_loss + alpha * worst_expert_loss
    else:
        loss = F.mse_loss(predictions, batch_Y)
    
    loss.backward()
    optimizer.step()
    return loss.item()


def evaluate_moe_model(A_data_tensor, S_data_tensor, Y_data_tensor, model, loss_fn, device):
    model.eval()
    with torch.no_grad():
        predictions = model(A_data_tensor.to(device), S_data_tensor.to(device))
        loss = loss_fn(predictions, Y_data_tensor.to(device))
    return loss.item()

def train_baseline_model(batch_X_combined, batch_Y, model, loss_fn, optimizer, device):
    model.train()
    batch_X_combined, batch_Y = batch_X_combined.to(device), batch_Y.to(device)
    
    optimizer.zero_grad()
    predictions = model(batch_X_combined)
    loss = loss_fn(predictions, batch_Y)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate_baseline_model(X_combined_tensor, Y_data_tensor, model, loss_fn, device):
    model.eval()
    with torch.no_grad():
        predictions = model(X_combined_tensor.to(device))
        loss = loss_fn(predictions, Y_data_tensor.to(device))
    return loss.item()

def train_rome_moe_with_dro(batch_A, batch_S, batch_Y, model, optimizer, device, alpha=0.1):
    model.train()
    batch_A, batch_S, batch_Y = batch_A.to(device), batch_S.to(device), batch_Y.to(device)
    
    optimizer.zero_grad()
    predictions = model(batch_A, batch_S)
    
    with torch.no_grad():
        if isinstance(model, FairMoE_S):
            gating_logits = model.gating_net(batch_S)
        elif isinstance(model, FairMoE_AS):
            combined_input = torch.cat([batch_A, batch_S], dim=-1)
            gating_logits = model.gating_net(combined_input)
        else:
            raise ValueError("Wrong model input!")
        
        group_assignments = torch.argmax(gating_logits, dim=1)
    
    group_losses = []
    for k in range(model.num_experts):
        mask = (group_assignments == k)
        if mask.sum() > 0:
            group_loss = F.mse_loss(predictions[mask], batch_Y[mask])
            group_losses.append(group_loss)
    
    if len(group_losses) > 0:
        avg_loss = F.mse_loss(predictions, batch_Y)
        worst_group_loss = torch.max(torch.stack(group_losses))
        loss = (1 - alpha) * avg_loss + alpha * worst_group_loss
    else:
        loss = F.mse_loss(predictions, batch_Y)
    
    loss.backward()
    optimizer.step()
    return loss.item()

def is_numeric_continuous(series, threshold_unique_ratio=0.1):

    if not pd.api.types.is_numeric_dtype(series):
        return False
    
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return False
    if clean_series.dtype in ['int64', 'int32'] and len(clean_series.unique()) <= 10:
        return False
    if clean_series.dtype in ['float64', 'float32']:
        return True

    unique_ratio = len(clean_series.unique()) / len(clean_series)
    return unique_ratio >= threshold_unique_ratio
