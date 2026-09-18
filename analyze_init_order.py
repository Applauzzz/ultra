#!/usr/bin/env python3
"""Analyze initialization order in train_eeg.py to find worker conflict."""

import re

# Read train_eeg.py
with open('/mnt/zehao/ultra/main/train_eeg.py', 'r') as f:
    content = f.read()

# Find the train function
train_func = re.search(r'def train\(args.*?\n(.*?)(?=\ndef |\Z)', content, re.DOTALL)
if not train_func:
    print("Could not find train function")
    exit(1)

train_body = train_func.group(1)

# Extract key initialization steps
steps = []
for line_num, line in enumerate(train_body.split('\n'), 1):
    line = line.strip()
    
    # Look for important initializations
    if any(keyword in line for keyword in [
        'logger.info',
        'wandb',
        'build_mae_model',
        'parallelize_model', 
        'build_optimizer',
        'build_eeg_dataloader',
        'iter(data_loader)',
        'checkpoint.load',
        'MetricLogger',
        'model.train()',
        'dist.barrier',
        'gc.disable',
    ]):
        # Extract the relevant part
        if 'logger.info' in line:
            msg = re.search(r'logger\.info\(["\']([^"\']+)', line)
            if msg:
                steps.append(f"Line {line_num}: LOG: {msg.group(1)}")
        elif 'wandb' in line and '=' in line:
            steps.append(f"Line {line_num}: WandB initialization")
        elif 'build_mae_model' in line:
            steps.append(f"Line {line_num}: Build model")
        elif 'parallelize_model' in line:
            steps.append(f"Line {line_num}: Parallelize model (FSDP/DDP)")
        elif 'build_optimizer' in line:
            steps.append(f"Line {line_num}: Build optimizer")
        elif 'build_eeg_dataloader' in line:
            steps.append(f"Line {line_num}: Build DataLoader")
        elif 'iter(data_loader)' in line or 'data_loader = iter' in line:
            steps.append(f"Line {line_num}: *** CREATE ITERATOR *** <-- LIKELY HANG POINT")
        elif 'checkpoint.load' in line:
            steps.append(f"Line {line_num}: Checkpoint load (has dist.barrier)")
        elif 'MetricLogger' in line:
            steps.append(f"Line {line_num}: Create MetricLogger")
        elif 'model.train()' in line:
            steps.append(f"Line {line_num}: Set model to train mode")
        elif 'gc.disable' in line:
            steps.append(f"Line {line_num}: Disable GC")

print("=" * 80)
print("INITIALIZATION ORDER IN train_eeg.py train() function")
print("=" * 80)
for step in steps:
    print(step)
print("=" * 80)
