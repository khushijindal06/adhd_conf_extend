
import time
import json
import statistics as stats

import torch
from torch.utils.flop_counter import FlopCounterMode

from model import ADHDNet, ADHDNetEnhanced

# ── Config ─────────────────────────────────────────────────────────────────
OUT_DIR  = '/kaggle/working'
N_WARMUP = 10
N_TIMED  = 50
BATCHES  = [1, 8]
SEED     = 42

torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {DEVICE}")
if DEVICE.type == 'cpu':
    torch.set_num_threads(torch.get_num_threads())
    print(f"CPU threads : {torch.get_num_threads()}")

MODELS = {
    'lightweight' : ADHDNet,
    'enhanced'    : ADHDNetEnhanced,
}

results = {}

for name, cls in MODELS.items():
    model = cls(n_pheno=5, dropout=0.3).to(DEVICE)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())

    for bs in BATCHES:
        vol   = torch.randn(bs, 1, 64, 64, 64, device=DEVICE)
        pheno = torch.randn(bs, 5, device=DEVICE)

        # ── FLOPs (hardware-independent) ──────────────────────────────────────
        with torch.no_grad():
            flop_counter = FlopCounterMode(display=False, depth=None)
            with flop_counter:
                _ = model(vol, pheno)
            total_flops = flop_counter.get_total_flops()

        # ── Latency ─────────────────────────────────────────────────────────
        with torch.no_grad():
            for _ in range(N_WARMUP):
                _ = model(vol, pheno)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()

            times_ms = []
            for _ in range(N_TIMED):
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = model(vol, pheno)
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000.0)

        mean_ms   = stats.mean(times_ms)
        std_ms    = stats.stdev(times_ms)
        median_ms = stats.median(times_ms)
        p95_ms    = sorted(times_ms)[int(0.95 * len(times_ms)) - 1]

        key = f"{name}_bs{bs}"
        results[key] = {
            'model'            : name,
            'batch_size'       : bs,
            'device'           : str(DEVICE),
            'params'           : n_params,
            'flops_total'      : total_flops,
            'flops_per_sample' : total_flops / bs,
            'gflops_per_sample': total_flops / bs / 1e9,
            'latency_mean_ms'  : mean_ms,
            'latency_std_ms'   : std_ms,
            'latency_median_ms': median_ms,
            'latency_p95_ms'   : p95_ms,
            'throughput_samples_per_s': bs / (mean_ms / 1000.0),
        }

        print(f"\n{name}  (batch={bs}, {n_params:,} params)")
        print(f"  FLOPs / sample : {total_flops/bs:,.0f}  ({total_flops/bs/1e9:.4f} GFLOPs)")
        print(f"  Latency        : {mean_ms:.2f} ± {std_ms:.2f} ms  "
              f"(median {median_ms:.2f}, p95 {p95_ms:.2f})")
        print(f"  Throughput     : {bs/(mean_ms/1000.0):.2f} samples/s")

with open(f'{OUT_DIR}/benchmark_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved → {OUT_DIR}/benchmark_results.json")
