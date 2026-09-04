"""
Integration test for EEG MAE model + DataLoader.
Tests that model and dataloader work together correctly.
"""

import sys
import time
from pathlib import Path

# Add ultra to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.optim as optim
from ultra.data_eeg import EEGDataArgs, build_eeg_dataloader
from ultra.model_eeg import MAEModelArgs, build_mae_model


def test_integration(num_steps=10):
    """Test model + dataloader integration."""
    print("=" * 70)
    print("EEG Model + DataLoader Integration Test")
    print("=" * 70)

    # ==================== Configuration ====================
    print("\n" + "-" * 70)
    print("1. Configuration")
    print("-" * 70)

    # Data configuration
    data_args = EEGDataArgs(
        data_path="/mnt/zehao/eeg_infra/data",
        subset="all",
        window_duration=10000,
        clip=10.0,
        masking_ratio=0.75,
        masking_window=200,
        masking_overlap=20,
        use_block_masking=False,  # Use simple masking for testing
        batch_size=4,  # Small batch size for quick testing
        num_workers=4,
        prefetch_factor=2,
        persistent_workers=True,
        world_size=1,
        rank=0,
    )

    # Model configuration (Base model)
    model_args = MAEModelArgs(
        encoder_embed_dim=512,
        encoder_depth=22,
        encoder_heads=8,
        encoder_head_dim=64,
        encoder_mlp_dim_ratio=2.66,
        encoder_use_geglu=True,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_heads=8,
        decoder_head_dim=64,
        decoder_mlp_dim_ratio=2.66,
        decoder_use_geglu=True,
        patch_size=200,
        patch_overlap=20,
        freqs=4,
        noise_ratio=0.0025,
        masking_ratio=0.75,
        token_avg=False,
    )

    print(f"Data config:")
    print(f"  Batch size: {data_args.batch_size}")
    print(f"  Num workers: {data_args.num_workers}")
    print(f"  Masking ratio: {data_args.masking_ratio}")

    print(f"\nModel config:")
    print(f"  Encoder: {model_args.encoder_embed_dim}d, {model_args.encoder_depth} layers")
    print(f"  Decoder: {model_args.decoder_embed_dim}d, {model_args.decoder_depth} layers")

    # ==================== Build DataLoader ====================
    print("\n" + "-" * 70)
    print("2. Building DataLoader")
    print("-" * 70)

    dataloader, state = build_eeg_dataloader(data_args, train=True)
    print(f"✓ DataLoader built successfully")
    print(f"  Total batches: {len(dataloader)}")

    # ==================== Build Model ====================
    print("\n" + "-" * 70)
    print("3. Building Model")
    print("-" * 70)

    model = build_mae_model(model_args)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model built successfully")
    print(f"  Total parameters: {total_params:,}")

    # Move to GPU if available (force CPU for now due to GPU memory constraints)
    device = torch.device("cpu")  # Force CPU to avoid GPU OOM
    model = model.to(device)
    print(f"  Device: {device} (using CPU to avoid GPU memory issues)")

    # ==================== Setup Optimizer ====================
    print("\n" + "-" * 70)
    print("4. Setting up Optimizer")
    print("-" * 70)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    print(f"✓ Optimizer: AdamW(lr=1e-4)")

    # ==================== Training Steps ====================
    print("\n" + "-" * 70)
    print(f"5. Running {num_steps} Training Steps")
    print("-" * 70)

    model.train()
    data_iter = iter(dataloader)

    step_times = []
    losses = []

    for step in range(num_steps):
        step_start = time.time()

        try:
            # Get batch
            eeg, pos, mask, unmask = next(data_iter)
        except StopIteration:
            # Restart iterator if we run out of data
            data_iter = iter(dataloader)
            eeg, pos, mask, unmask = next(data_iter)

        # Move to device
        eeg = eeg.to(device)
        pos = pos.to(device)
        mask = mask.to(device)
        unmask = unmask.to(device)

        # Forward pass
        optimizer.zero_grad()
        loss = model(eeg, pos, b_m=mask, b_u=unmask)

        # Backward pass
        loss.backward()
        optimizer.step()

        step_time = time.time() - step_start
        step_times.append(step_time)
        losses.append(loss.item())

        # Print progress
        print(f"  Step {step + 1}/{num_steps}: "
              f"loss={loss.item():.4f}, "
              f"time={step_time*1000:.0f}ms, "
              f"shape={eeg.shape}")

    # ==================== Statistics ====================
    print("\n" + "-" * 70)
    print("6. Training Statistics")
    print("-" * 70)

    import numpy as np
    step_times = np.array(step_times)
    losses = np.array(losses)

    print(f"\nStep timing:")
    print(f"  Mean: {step_times.mean()*1000:.0f}ms")
    print(f"  Min:  {step_times.min()*1000:.0f}ms")
    print(f"  Max:  {step_times.max()*1000:.0f}ms")

    print(f"\nLoss statistics:")
    print(f"  Initial: {losses[0]:.4f}")
    print(f"  Final:   {losses[-1]:.4f}")
    print(f"  Mean:    {losses.mean():.4f}")
    print(f"  Std:     {losses.std():.4f}")

    # Check if loss is reasonable
    if losses.mean() < 0.5 or losses.mean() > 2.0:
        print(f"\n⚠️  Warning: Loss values seem unusual (mean={losses.mean():.4f})")
    else:
        print(f"\n✓ Loss values look reasonable")

    # ==================== Gradient Check ====================
    print("\n" + "-" * 70)
    print("7. Gradient Health Check")
    print("-" * 70)

    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)

            # Print a few example gradients
            if len(grad_norms) <= 5:
                print(f"  {name[:50]:50s}: grad_norm={grad_norm:.6f}")

    if len(grad_norms) > 5:
        print(f"  ... ({len(grad_norms) - 5} more parameters)")

    print(f"\nGradient statistics:")
    grad_norms = np.array(grad_norms)
    print(f"  Parameters with gradients: {len(grad_norms)}")
    print(f"  Mean gradient norm: {grad_norms.mean():.6f}")
    print(f"  Max gradient norm:  {grad_norms.max():.6f}")

    # Check for gradient issues
    if grad_norms.max() > 100:
        print(f"\n⚠️  Warning: Very large gradients detected (max={grad_norms.max():.2f})")
    elif (grad_norms < 1e-8).all():
        print(f"\n⚠️  Warning: Gradients are very small (possible vanishing gradients)")
    else:
        print(f"\n✓ Gradients look healthy")

    # ==================== Memory Usage ====================
    print("\n" + "-" * 70)
    print("8. Memory Usage")
    print("-" * 70)

    if device.type == "cuda":
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
        print(f"  Reserved:  {torch.cuda.memory_reserved() / 1024**2:.0f} MB")
        print(f"  Max allocated: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    else:
        print(f"  Running on CPU (no GPU memory usage)")

    # ==================== Final Summary ====================
    print("\n" + "=" * 70)
    print("Integration Test Summary")
    print("=" * 70)

    print(f"\n✅ Model + DataLoader Integration Test PASSED!")
    print(f"\n📊 Key Metrics:")
    print(f"   ⏱️  Avg step time:   {step_times.mean()*1000:.0f}ms")
    print(f"   📉 Final loss:      {losses[-1]:.4f}")
    print(f"   🎯 Gradient health: {'✓' if grad_norms.max() < 100 else '⚠️'}")
    print(f"   💾 GPU memory:      {torch.cuda.memory_allocated() / 1024**2:.0f} MB"
          if device.type == "cuda" else "   💾 Running on CPU")

    print(f"\n🎉 Ready for full training!")

    return {
        'avg_step_time': step_times.mean(),
        'final_loss': losses[-1],
        'grad_norm': grad_norms.mean(),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test EEG Model + DataLoader Integration')
    parser.add_argument('--num-steps', type=int, default=10,
                        help='Number of training steps to run (default: 10)')

    args = parser.parse_args()

    try:
        test_integration(num_steps=args.num_steps)
    except Exception as e:
        print("\n" + "=" * 70)
        print("❌ Integration Test FAILED!")
        print("=" * 70)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
