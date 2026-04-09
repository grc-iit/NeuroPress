#!/usr/bin/env python3
"""
Fine-tune a pretrained ViT model and export training checkpoints as raw .f32 files.

Produces evolving checkpoint data at multiple training stages, with 4 tensor
types per checkpoint: weights, adam_m (1st moment), adam_v (2nd moment), gradients.
Each .f32 file is a flat float32 array, compatible with the generic_benchmark
generic_benchmark.cu driver.

Files are named epoch-major (epoch01_adam_m.f32, epoch01_adam_v.f32, etc.)
so alphabetical sort = epoch order for NN-RL learning.

Usage:
    # Full ViT-Large run (~45-60 min on A100, generates ~37 GB)
    python3 scripts/train_and_export_checkpoints.py

    # Quick test with ViT-Base (~10 min, ~2.6 GB)
    python3 scripts/train_and_export_checkpoints.py --model vit_b_16 --epochs 5 --checkpoint-epochs 1,3,5

    # Custom output
    python3 scripts/train_and_export_checkpoints.py --outdir /tmp/ckpt --epochs 10
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T


def compute_dims_2d(n_elements):
    """Find a 2D factorization for generic_benchmark --dims. Returns (d0, d1) with d0*d1 >= n_elements."""
    # Try exact factorization first
    s = int(math.isqrt(n_elements))
    while s > 1 and n_elements % s != 0:
        s -= 1
    if n_elements % s == 0:
        return (s, n_elements // s)
    # No clean factor — pad to next value divisible by a reasonable factor
    target = n_elements
    while True:
        s = int(math.isqrt(target))
        while s > 1 and target % s != 0:
            s -= 1
        if s > 1:
            return (s, target // s)
        target += 1


def export_tensor_padded(tensor_list, path, target_elements):
    """Concatenate parameter tensors, pad to target_elements, save as .f32."""
    flat = torch.cat([p.detach().float().cpu().flatten() for p in tensor_list])
    n = flat.numel()
    if n < target_elements:
        flat = torch.cat([flat, torch.zeros(target_elements - n)])
    elif n > target_elements:
        flat = flat[:target_elements]
    flat.numpy().tofile(path)
    return os.path.getsize(path)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune ViT and export training checkpoints as .f32")
    parser.add_argument("--model", default="vit_l_16",
                        choices=["vit_l_16", "vit_b_16", "resnet18"],
                        help="Model: vit_l_16 (304M), vit_b_16 (86M), resnet18 (11M, quick test)")
    parser.add_argument("--dataset", default="cifar10",
                        help="Dataset (default: cifar10)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Total training epochs (default: 20)")
    parser.add_argument("--checkpoint-epochs", type=str, default="1,2,3,5,8,10,15,20",
                        help="Comma-separated epochs to export (default: 1,2,3,5,8,10,15,20)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Training batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--outdir", type=str, default=None,
                        help="Output directory (default: data/ai_training/vit_{model}_{dataset})")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Dataset cache root (default: data/)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true",
                        help="Use automatic mixed precision (faster training)")
    parser.add_argument("--hdf5-direct", action="store_true",
                        help="Write checkpoints directly from GPU via HDF5 VOL (no CPU roundtrip)")
    parser.add_argument("--chunk-mb", type=int, default=4,
                        help="HDF5 chunk size in MB for --hdf5-direct (default: 4)")
    parser.add_argument("--error-bound", type=float, default=0.0,
                        help="Lossy error bound for --hdf5-direct (default: 0.0 = lossless)")
    parser.add_argument("--policy", type=str, default="balanced",
                        choices=["balanced", "ratio", "speed"],
                        help="NN cost model policy for --hdf5-direct (default: balanced)")
    parser.add_argument("--benchmark", action="store_true",
                        help="Benchmark all compression algorithms at each checkpoint (requires --hdf5-direct)")
    parser.add_argument("--benchmark-configs", type=str, default=None,
                        help="Multi-config benchmark: 'chunk_mb:error_bound:outdir,...' "
                             "Train once, benchmark each config at every checkpoint. "
                             "Example: '4:0.0:out/4mb_ll,16:0.0:out/16mb_ll,4:0.01:out/4mb_lossy'")
    parser.add_argument("--sgd-lr", type=float, default=0.2,
                        help="SGD learning rate for online NN updates (default: 0.2)")
    parser.add_argument("--sgd-mape", type=float, default=0.10,
                        help="MAPE threshold to trigger SGD update (default: 0.10)")
    parser.add_argument("--explore-k", type=int, default=4,
                        help="Number of exploration alternatives (default: 4)")
    parser.add_argument("--explore-thresh", type=float, default=0.20,
                        help="Cost error threshold for exploration (default: 0.20)")
    parser.add_argument("--max-batches-per-epoch", type=int, default=None,
                        help="Limit training to N batches per epoch (makes training dumber/faster, "
                             "increases I/O fraction). Default: use full dataset.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip validation loop each epoch (faster, removes ~3s/epoch overhead).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_epochs = sorted(set(int(e) for e in args.checkpoint_epochs.split(",")))
    if max(checkpoint_epochs) > args.epochs:
        print(f"Warning: max checkpoint epoch {max(checkpoint_epochs)} > total epochs {args.epochs}")
        checkpoint_epochs = [e for e in checkpoint_epochs if e <= args.epochs]

    # Resolve paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_root = args.data_root or os.path.join(project_dir, "data")
    model_short = args.model.replace("vit_", "vit")  # vit_l_16 → vitl16
    if args.outdir:
        outdir = args.outdir
    else:
        if args.model.startswith("vit_"):
            dir_name = f"vit_{args.model.split('_')[1]}_{args.dataset}"
        else:
            dir_name = f"{args.model}_{args.dataset}"
        outdir = os.path.join(project_dir, "data", "ai_training", dir_name)
    os.makedirs(outdir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── HDF5 direct writer (optional) ──
    hdf5_writer = None
    if args.hdf5_direct:
        from gpucompress_hdf5 import GPUCompressHDF5Writer, concat_and_pad_gpu
        weights_path = os.path.join(project_dir, "neural_net", "weights", "model.nnwt")
        if not os.path.exists(weights_path):
            weights_path = None
        hdf5_writer = GPUCompressHDF5Writer(
            lib_dir=os.path.join(project_dir, "build"),
            weights_path=weights_path,
        )
        hdf5_writer.init()
        hdf5_writer.set_policy(args.policy)
        print(f"  HDF5 direct write enabled (chunk={args.chunk_mb}MB, eb={args.error_bound}, policy={args.policy})")

    # ── Load model ──
    print(f"Loading pretrained {args.model}...")
    num_classes = 10
    if args.model == "vit_l_16":
        weights = torchvision.models.ViT_L_16_Weights.DEFAULT
        model = torchvision.models.vit_l_16(weights=weights)
        model.heads.head = nn.Linear(1024, num_classes)
    elif args.model == "vit_b_16":
        weights = torchvision.models.ViT_B_16_Weights.DEFAULT
        model = torchvision.models.vit_b_16(weights=weights)
        model.heads.head = nn.Linear(768, num_classes)
    elif args.model == "resnet18":
        weights = torchvision.models.ResNet18_Weights.DEFAULT
        model = torchvision.models.resnet18(weights=weights)
        model.fc = nn.Linear(512, num_classes)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Unfreeze ALL layers (full fine-tuning)
    for p in model.parameters():
        p.requires_grad = True

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    param_mb = n_params * 4 / (1024 * 1024)

    # Compute padded dims
    d0, d1 = compute_dims_2d(n_params)
    target_elements = d0 * d1
    pad_elements = target_elements - n_params

    print(f"  Model       : {args.model}")
    print(f"  Parameters  : {n_params:,} ({param_mb:.1f} MB as float32)")
    print(f"  Padded dims : {d0} x {d1} = {target_elements:,} ({pad_elements} padding zeros)")
    print(f"  Device      : {device}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Checkpoints : {checkpoint_epochs}")
    print(f"  Output      : {outdir}")
    print()

    # ── Inline full benchmark (optional) ──
    inline_bench = None
    bench_configs = []  # list of (chunk_mb, error_bound, outdir, bench_csv, chunk_csv, InlineFullBenchmark)
    if hdf5_writer is not None and (args.benchmark or args.benchmark_configs):
        from gpucompress_hdf5 import InlineFullBenchmark
        nn_weights = os.path.join(project_dir, "neural_net", "weights", "model.nnwt")

        if args.benchmark_configs:
            # Multi-config: train once, benchmark multiple chunk/eb combos
            for cfg_str in args.benchmark_configs.split(","):
                parts = cfg_str.strip().split(":")
                if len(parts) != 3:
                    print(f"  WARNING: invalid benchmark config '{cfg_str}', expected chunk_mb:eb:outdir")
                    continue
                c_mb, c_eb, c_outdir = int(parts[0]), float(parts[1]), parts[2]
                os.makedirs(c_outdir, exist_ok=True)
                bench = InlineFullBenchmark(hdf5_writer, nn_weights, target_elements,
                                           sgd_lr=args.sgd_lr, sgd_mape=args.sgd_mape,
                                           explore_k=args.explore_k, explore_thresh=args.explore_thresh)
                bench_configs.append({
                    "chunk_mb": c_mb, "error_bound": c_eb, "outdir": c_outdir,
                    "bench": bench, "bench_csv": None, "chunk_csv": None,
                })
            print(f"  Multi-config benchmark: {len(bench_configs)} configs × 15 algorithms")
            print(f"  SGD: lr={args.sgd_lr}, mape={args.sgd_mape} | Explore: k={args.explore_k}, thresh={args.explore_thresh}")
            for bc in bench_configs:
                mode = "lossless" if bc["error_bound"] == 0.0 else f"lossy(eb={bc['error_bound']})"
                print(f"    {bc['chunk_mb']}MB {mode} → {bc['outdir']}")
        else:
            # Single-config benchmark (original --benchmark behavior)
            inline_bench = InlineFullBenchmark(hdf5_writer, nn_weights, target_elements,
                                               sgd_lr=args.sgd_lr, sgd_mape=args.sgd_mape,
                                               explore_k=args.explore_k, explore_thresh=args.explore_thresh)
            print(f"  Inline benchmark: 15 configs (9 fixed + 6 NN)")
            print(f"  SGD: lr={args.sgd_lr}, mape={args.sgd_mape} | Explore: k={args.explore_k}, thresh={args.explore_thresh}")

    # ── Dataset ──
    transform_train = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform_val = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading CIFAR-10 (auto-downloads if needed)...")
    cifar_root = os.path.join(data_root, "cifar10")
    train_ds = torchvision.datasets.CIFAR10(root=cifar_root, train=True,
                                             download=True, transform=transform_train)
    val_ds = torchvision.datasets.CIFAR10(root=cifar_root, train=False,
                                           download=True, transform=transform_val)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    steps_per_epoch = len(train_loader)
    print(f"  Train: {len(train_ds)} images, {steps_per_epoch} steps/epoch")
    print(f"  Val  : {len(val_ds)} images")
    print()

    # ── Optimizer + scheduler ──
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=args.amp)

    # ── Training loop ──
    print("=" * 60)
    print("  Training")
    print("=" * 60)

    total_start = time.time()
    bench_csv = None
    chunk_csv = None

    # GPU warmup: run one forward pass to trigger CUDA JIT kernel compilation
    # before starting the e2e timer, so first-epoch compilation overhead is excluded.
    if hdf5_writer is not None:
        model.eval()
        with torch.no_grad():
            _warmup_images, _ = next(iter(train_loader))
            _warmup_images = _warmup_images.to(device)
            _ = model(_warmup_images)
        torch.cuda.synchronize()
        model.train()

    # Reset e2e timer AFTER warmup to exclude model loading, CUDA init, and data prep.
    # This makes e2e_ms reflect only the active training + I/O time.
    if hdf5_writer is not None:
        hdf5_writer.record_process_start()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_start = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            if args.max_batches_per_epoch is not None and batch_idx >= args.max_batches_per_epoch:
                break

            images, labels = images.to(device), labels.to(device)

            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            epoch_correct += predicted.eq(labels).sum().item()
            epoch_total += images.size(0)

            if (batch_idx + 1) % 50 == 0:
                pct = 100.0 * (batch_idx + 1) / steps_per_epoch
                sys.stdout.write(f"\r  Epoch {epoch:2d}/{args.epochs} "
                                 f"[{pct:5.1f}%] loss={loss.item():.4f}")
                sys.stdout.flush()

        scheduler.step()

        epoch_time = time.time() - epoch_start
        train_acc = 100.0 * epoch_correct / epoch_total
        avg_loss = epoch_loss / epoch_total

        # Quick validation (skip if --no-validate)
        val_acc = float("nan")
        if not args.no_validate:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                        outputs = model(images)
                    _, predicted = outputs.max(1)
                    val_correct += predicted.eq(labels).sum().item()
                    val_total += images.size(0)
            val_acc = 100.0 * val_correct / val_total

        print(f"\r  Epoch {epoch:2d}/{args.epochs}  "
              f"loss={avg_loss:.4f}  train_acc={train_acc:.1f}%  "
              f"val_acc={val_acc:.1f}%  time={epoch_time:.1f}s  "
              f"lr={scheduler.get_last_lr()[0]:.2e}")

        # ── Export checkpoint if this is a checkpoint epoch ──
        if epoch in checkpoint_epochs:
            print(f"\n  >>> Exporting checkpoint at epoch {epoch}...")
            export_start = time.time()

            # Helper: collect Adam state tensors
            def _adam_tensors(key):
                tensors = []
                for group in optimizer.param_groups:
                    for p in group["params"]:
                        if p in optimizer.state and key in optimizer.state[p]:
                            tensors.append(optimizer.state[p][key])
                        else:
                            tensors.append(torch.zeros_like(p))
                return tensors

            # Compute gradients for this checkpoint
            model.train()
            optimizer.zero_grad()
            images, labels = next(iter(train_loader))
            images, labels = images.to(device), labels.to(device)
            with torch.amp.autocast(device_type="cuda", enabled=args.amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            grad_tensors = [p.grad if p.grad is not None else torch.zeros_like(p)
                            for p in model.parameters()]

            # Export 4 tensor types
            tensor_sets = [
                ("weights",   list(model.parameters())),
                ("adam_m",    _adam_tensors("exp_avg")),
                ("adam_v",    _adam_tensors("exp_avg_sq")),
                ("gradients", grad_tensors),
            ]

            CSV_HEADER = ("epoch,tensor,algorithm,policy,mode,ratio,"
                         "write_ms,read_ms,write_mbps,read_mbps,"
                         "file_bytes,orig_bytes,mismatches,"
                         "n_chunks,sgd_fires,explorations,"
                         "mape_ratio_pct,mape_comp_pct,mape_decomp_pct,mape_psnr_pct,"
                         "mae_ratio,mae_comp_ms,mae_decomp_ms,mae_psnr_db,"
                         "r2_ratio,r2_comp,r2_decomp,r2_psnr,"
                         "nn_ms,stats_ms,preproc_ms,comp_ms,decomp_ms,"
                         "explore_ms,sgd_ms,"
                         "stage1_ms,drain_ms,io_drain_ms,pipeline_ms,"
                         "s2_busy_ms,s3_busy_ms,"
                         "psnr_db,rmse,max_abs_err,mean_abs_err,ssim,bit_rate,data_range\n")
            CHUNK_CSV_HEADER = ("epoch,tensor,algorithm,policy,mode,"
                                "chunk_idx,action,actual_ratio,predicted_ratio,"
                                "comp_ms,predicted_comp_time,"
                                "decomp_ms,predicted_decomp_time,"
                                "predicted_psnr,actual_psnr,"
                                "sgd_fired,exploration_triggered\n")

            # Open benchmark CSVs on first checkpoint
            if inline_bench is not None and bench_csv is None:
                bench_csv = open(os.path.join(outdir, "inline_benchmark.csv"), "w")
                bench_csv.write(CSV_HEADER)
                chunk_csv = open(os.path.join(outdir, "inline_benchmark_chunks.csv"), "w")
                chunk_csv.write(CHUNK_CSV_HEADER)

            # Open multi-config CSVs on first checkpoint
            if bench_configs:
                for bc in bench_configs:
                    if bc["bench_csv"] is None:
                        bc["bench_csv"] = open(os.path.join(bc["outdir"], "inline_benchmark.csv"), "w")
                        bc["bench_csv"].write(CSV_HEADER)
                        bc["chunk_csv"] = open(os.path.join(bc["outdir"], "inline_benchmark_chunks.csv"), "w")
                        bc["chunk_csv"].write(CHUNK_CSV_HEADER)

            for name, tensors in tensor_sets:
                if hdf5_writer is not None or bench_configs or inline_bench is not None:
                    flat = concat_and_pad_gpu(tensors, target_elements)
                else:
                    flat = None

                if bench_configs:
                    # Multi-config: benchmark same tensor with each config
                    for bc in bench_configs:
                        mode_label = "lossless" if bc["error_bound"] == 0.0 else f"lossy(eb={bc['error_bound']})"
                        print(f"      [{bc['chunk_mb']}MB {mode_label}]")
                        bc["bench"].run_checkpoint(
                            hdf5_writer, flat, target_elements,
                            epoch, name, bc["outdir"],
                            chunk_elements=bc["chunk_mb"] * 1024 * 1024 // 4,
                            error_bound=bc["error_bound"],
                            csv_file=bc["bench_csv"],
                            chunk_csv_file=bc["chunk_csv"],
                        )
                    # Save checkpoint with default config
                    h5_path = os.path.join(outdir, f"epoch{epoch:02d}_{name}.h5")
                    hdf5_writer.write_gpu_tensor(
                        flat.data_ptr(), target_elements, h5_path, "data",
                        chunk_elements=args.chunk_mb * 1024 * 1024 // 4,
                        error_bound=args.error_bound,
                    )
                elif inline_bench is not None:
                    # Single-config benchmark
                    inline_bench.run_checkpoint(
                        hdf5_writer, flat, target_elements,
                        epoch, name, outdir,
                        chunk_elements=args.chunk_mb * 1024 * 1024 // 4,
                        error_bound=args.error_bound,
                        csv_file=bench_csv,
                        chunk_csv_file=chunk_csv,
                    )
                    h5_path = os.path.join(outdir, f"epoch{epoch:02d}_{name}.h5")
                    hdf5_writer.write_gpu_tensor(
                        flat.data_ptr(), target_elements, h5_path, "data",
                        chunk_elements=args.chunk_mb * 1024 * 1024 // 4,
                        error_bound=args.error_bound,
                    )
                elif hdf5_writer is not None:
                    h5_path = os.path.join(outdir, f"epoch{epoch:02d}_{name}.h5")
                    hdf5_writer.write_gpu_tensor(
                        flat.data_ptr(), target_elements, h5_path, "data",
                        chunk_elements=args.chunk_mb * 1024 * 1024 // 4,
                        error_bound=args.error_bound,
                    )
                    sz = os.path.getsize(h5_path)
                    print(f"      epoch{epoch:02d}_{name}.h5  {sz/1024/1024:>7.1f} MB")
                else:
                    f32_path = os.path.join(outdir, f"epoch{epoch:02d}_{name}.f32")
                    export_tensor_padded(tensors, f32_path, target_elements)
                    sz = os.path.getsize(f32_path)
                    print(f"      epoch{epoch:02d}_{name}.f32  {sz/1024/1024:>7.1f} MB")

                del flat
                torch.cuda.empty_cache()

            optimizer.zero_grad()  # Clean up gradients
            export_time = time.time() - export_start
            print(f"      Export time: {export_time:.1f}s\n")

    total_time = time.time() - total_start

    # ── Cleanup ──
    if bench_csv is not None:
        bench_csv.close()
    if chunk_csv is not None:
        chunk_csv.close()
    for bc in bench_configs:
        if bc.get("bench_csv"):
            bc["bench_csv"].close()
        if bc.get("chunk_csv"):
            bc["chunk_csv"].close()
    if hdf5_writer is not None:
        hdf5_writer.dump_timing()   # explicit flush before cleanup (atexit unreliable in ctypes)
        hdf5_writer.cleanup()

    # ── Auto-generate plots ──
    plot_dirs = []
    if bench_csv is not None:
        plot_dirs.append(outdir)
    for bc in bench_configs:
        plot_dirs.append(bc["outdir"])
    for pdir in plot_dirs:
        csv_path = os.path.join(pdir, "inline_benchmark.csv")
        if os.path.exists(csv_path):
            print(f"\n  Generating plots for {os.path.basename(pdir)}...")
            try:
                from plot_inline_benchmark import main as plot_main
                sys.argv = ["plot_inline_benchmark.py", csv_path]
                plot_main()
            except Exception as e:
                print(f"    Plot generation failed: {e}")

    # ── Write metadata ──
    _ext = ".h5" if args.hdf5_direct else ".f32"
    dims_str = f"{d0},{d1}"
    meta_path = os.path.join(outdir, "README.txt")
    with open(meta_path, "w") as f:
        f.write(f"ViT Training Checkpoint Data for GPUCompress Benchmarks\n")
        f.write(f"Generated by: scripts/train_and_export_checkpoints.py\n\n")
        f.write(f"Model: {args.model} ({n_params:,} parameters, {param_mb:.1f} MB)\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Epochs: {args.epochs}, checkpoints at: {checkpoint_epochs}\n")
        f.write(f"Optimizer: AdamW (lr={args.lr}, wd={args.weight_decay})\n")
        f.write(f"Dims for generic_benchmark: --dims {dims_str}\n")
        f.write(f"Padded elements: {target_elements:,} ({pad_elements} zeros appended)\n")
        f.write(f"Total training time: {total_time:.0f}s\n\n")
        f.write(f"Files:\n")
        for fname in sorted(os.listdir(outdir)):
            if fname.endswith(_ext):
                fsize = os.path.getsize(os.path.join(outdir, fname))
                f.write(f"  {fname:40s} {fsize/1024/1024:>8.1f} MB\n")
        f.write(f"\nUsage with benchmark:\n")
        f.write(f"  BENCHMARKS=ai_training AI_MODEL={args.model} \\\n")
        f.write(f"    CHUNK_MB=4 POLICIES=balanced VERIFY=0 bash benchmarks/benchmark.sh\n")

    # ── Summary ──
    n_files = len([f for f in os.listdir(outdir) if f.endswith(_ext)])
    total_disk = sum(os.path.getsize(os.path.join(outdir, f))
                     for f in os.listdir(outdir) if f.endswith(_ext))

    print()
    print("=" * 60)
    print(f"  Checkpoint Export Complete")
    print("=" * 60)
    print(f"  Model         : {args.model} ({n_params:,} params)")
    print(f"  Training time : {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Final val_acc : {val_acc:.1f}%")
    print(f"  Files         : {n_files} {_ext} files")
    print(f"  Total disk    : {total_disk/1024/1024/1024:.1f} GB")
    print(f"  Dims          : --dims {dims_str}")
    print(f"  Output        : {outdir}")
    print()
    print(f"  Add to run_sdr.sh:")
    if args.model.startswith("vit_"):
        ds_name = f"vit_{args.model.split('_')[1]}_{args.dataset}"
    else:
        ds_name = f"{args.model}_{args.dataset}"
    print(f'    DS_SUBDIR[{ds_name}]="{os.path.basename(outdir)}"')
    print(f'    DS_DIMS[{ds_name}]="{dims_str}"')
    print(f'    DS_EXT[{ds_name}]="{_ext}"')
    print()
    print(f"  Run benchmark:")
    print(f"    BENCHMARKS=ai_training AI_MODEL={ds_name} \\")
    print(f"      CHUNK_MB=4 POLICIES=balanced VERIFY=0 bash benchmarks/benchmark.sh")


if __name__ == "__main__":
    main()
    # Force clean exit when HDF5-direct was used to avoid atexit segfault
    # from CUDA/HDF5 library teardown order conflict
    # Always force clean exit when GPU libraries were loaded
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
