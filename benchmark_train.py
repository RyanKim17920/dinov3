# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import argparse
import json
import logging
import math
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from threading import Event, Thread
from typing import Optional, Tuple

import torch

try:
    import pynvml
    HAS_PYNVML = True
except ImportError:
    HAS_PYNVML = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import resource
except ImportError:
    resource = None

logger = logging.getLogger("dinov3_bench")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
BENCHMARK_VERSION = "3.1"
MIN_SAMPLES = 5
MIN_SAMPLES_FOR_CI = 5
DEFAULT_BATCH_SIZE = 64
DEFAULT_SKIP_TRANSIENT = 2
DEFAULT_TELEMETRY_PERIOD_SUBPROCESS = 1.0


class TelemetrySampler:
    """GPU/CPU telemetry sampler with pynvml support and subprocess fallback."""

    def __init__(self, sample_period_s: float = 0.25, use_pynvml: Optional[bool] = None, telemetry_backend: Optional[str] = None):
        self.sample_period_s = sample_period_s
        self._stop = Event()
        self._thread: Thread | None = None
        self._gpu_samples: list[dict[str, float]] = []
        self._cpu_samples: list[dict[str, float]] = []
        self._gpu_ids: list[int] = []
        self._pynvml_devices: list = []
        self._backend: str = ""
        self._nvidia_smi_missing = False
        self._torch_nvml_bad = False
        self._torch_empty_count = 0

        # Determine backend
        if telemetry_backend is not None:
            self._backend = telemetry_backend
        elif use_pynvml is not None:
            self._backend = "pynvml" if use_pynvml else "subprocess"
        else:
            if HAS_PYNVML:
                self._backend = "pynvml"
            elif torch.cuda.is_available():
                self._backend = "torch_cuda_nvml"
            else:
                self._backend = "subprocess"

        if self._backend == "pynvml":
            if not HAS_PYNVML:
                logger.warning("pynvml requested but not available; falling back to subprocess")
                self._backend = "subprocess"
            else:
                self._init_pynvml()

        logger.info(f"Telemetry backend: {self._backend} (period={sample_period_s}s)")

    def _init_pynvml(self):
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            self._gpu_ids.append(i)
            self._pynvml_devices.append(handle)
            name = pynvml.nvmlDeviceGetName(handle)
            logger.info(f"  pynvml GPU {i}: {name}")

    def start(self):
        if self._thread is not None:
            return
        self._thread = Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._backend == "pynvml":
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    def _sample_cpu(self):
        if resource is None:
            return None
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = usage.ru_maxrss / 1024.0 if os.name == "posix" else usage.ru_maxrss / (1024.0**2)
        return {
            "time": time.perf_counter(),
            "cpu_seconds": float(usage.ru_utime + usage.ru_stime),
            "rss_mb": float(rss_mb),
        }

    def _sample_gpu_pynvml(self) -> list[dict[str, float]]:
        samples = []
        for idx, handle in zip(self._gpu_ids, self._pynvml_devices):
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW -> W
                samples.append({
                    "time": time.perf_counter(),
                    "gpu_id": idx,
                    "gpu_util_pct": float(util.gpu),
                    "gpu_mem_util_pct": float(util.memory),
                    "gpu_mem_used_mb": float(mem.used) / (1024.0**2),
                    "gpu_mem_total_mb": float(mem.total) / (1024.0**2),
                    "gpu_power_w": float(power),
                })
            except Exception:
                pass
        return samples

    def _sample_gpu_torch_nvml(self) -> list[dict[str, float]]:
        """Sample via torch.cuda.nvml — avoids subprocess NVML lock contention during active training."""
        samples = []
        t = time.perf_counter()
        for idx in range(torch.cuda.device_count()):
            try:
                util = torch.cuda.nvml.device(idx).query_utilization()
                mem = torch.cuda.nvml.device(idx).query_memory_info()
                samples.append({
                    "time": t,
                    "gpu_id": idx,
                    "gpu_util_pct": float(util.gpu),
                    "gpu_mem_util_pct": float(util.memory),
                    "gpu_mem_used_mb": float(mem.used) / (1024.0**2),
                    "gpu_mem_total_mb": float(mem.total) / (1024.0**2),
                })
            except Exception:
                pass
        return samples

    def _sample_gpu_subprocess(self) -> list[dict[str, float]]:
        cmd = [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            if not self._nvidia_smi_missing:
                logger.warning("Telemetry: nvidia-smi not found in PATH — GPU telemetry will be NULL")
                self._nvidia_smi_missing = True
            return []
        except subprocess.TimeoutExpired:
            return []
        except Exception as exc:
            logger.warning(f"Telemetry subprocess error: {exc}")
            return []

        if result.returncode != 0:
            stderr_preview = (result.stderr or "")[:200]
            logger.warning(f"Telemetry: nvidia-smi failed (rc={result.returncode}): {stderr_preview}")
            return []

        samples = []
        t = time.perf_counter()
        for line in result.stdout.strip().splitlines():
            if not line.strip():
                continue
            tokens = [tok.strip() for tok in line.split(",")]
            if len(tokens) != 5:
                continue
            try:
                samples.append({
                    "time": t,
                    "gpu_id": int(tokens[0]),
                    "gpu_util_pct": float(tokens[1]),
                    "gpu_mem_util_pct": float(tokens[2]),
                    "gpu_mem_used_mb": float(tokens[3]),
                    "gpu_mem_total_mb": float(tokens[4]),
                })
            except ValueError:
                continue
        return samples

    def _check_torch_nvml_fallback(self, sample_batch: list[dict[str, float]]) -> None:
        """Detect if torch.cuda.nvml is returning parent-process-only stats (near-zero)
        or failing entirely (empty batch), and fall back to subprocess nvidia-smi."""
        if self._torch_nvml_bad:
            return
        if sample_batch:
            mem_utils = [s.get("gpu_mem_util_pct", 0) for s in sample_batch]
            if mem_utils and all(mu < 5.0 for mu in mem_utils):
                self._torch_nvml_bad = True
                logger.warning("torch.cuda.nvml reports near‑zero GPU mem util — falling back to subprocess (nvidia-smi)")
                new_batch = self._sample_gpu_subprocess()
                sample_batch.clear()
                sample_batch.extend(new_batch)
        else:
            self._torch_empty_count += 1
            if self._torch_empty_count >= 3:
                self._torch_nvml_bad = True
                logger.warning("torch.cuda.nvml repeatedly returned empty results — falling back to subprocess (nvidia-smi)")
                new_batch = self._sample_gpu_subprocess()
                sample_batch.clear()
                sample_batch.extend(new_batch)

    def _loop(self):
        while not self._stop.is_set():
            cpu_sample = self._sample_cpu()
            if cpu_sample is not None:
                self._cpu_samples.append(cpu_sample)

            if self._backend == "pynvml":
                for s in self._sample_gpu_pynvml():
                    self._gpu_samples.append(s)
            else:
                # Prefer torch.cuda.nvml (avoids subprocess NVML lock contention)
                sample_batch: list[dict[str, float]] = []
                if self._backend == "subprocess":
                    sample_batch = self._sample_gpu_subprocess()
                elif self._torch_nvml_bad:
                    sample_batch = self._sample_gpu_subprocess()
                elif torch.cuda.is_available():
                    sample_batch = self._sample_gpu_torch_nvml()
                # Fall back to subprocess if torch path returned nothing OR if values
                # look wrong (e.g., parent process with no CUDA context reports near‑zero
                # memory; subprocess gives device‑wide stats regardless of context).
                if not sample_batch:
                    sample_batch = self._sample_gpu_subprocess()
                if self._backend == "torch_cuda_nvml":
                    self._check_torch_nvml_fallback(sample_batch)
                for s in sample_batch:
                    self._gpu_samples.append(s)

            self._stop.wait(self.sample_period_s)

    def summary(self, start_wall_s: float, end_wall_s: float) -> dict:
        wall_time = max(end_wall_s - start_wall_s, 0.0)

        backend_label = self._backend
        if self._torch_nvml_bad:
            backend_label = "torch_cuda_nvml (→ subprocess fallback)"
        summary = {
            "backend": backend_label,
            "wall_time_s": wall_time,
            "cpu_seconds": None,
            "cpu_util_pct": None,
            "rss_mb": None,
            "gpu_util_pct": None,
            "gpu_mem_util_pct": None,
            "gpu_mem_used_mb": None,
            "gpu_mem_total_mb": None,
            "per_gpu": {},
        }

        if self._cpu_samples:
            cpu_used = self._cpu_samples[-1]["cpu_seconds"] - self._cpu_samples[0]["cpu_seconds"]
            peak_rss = max(s["rss_mb"] for s in self._cpu_samples)
            summary["cpu_seconds"] = cpu_used
            summary["rss_mb"] = peak_rss
            if wall_time > 0 and (c := os.cpu_count()) > 0:
                summary["cpu_util_pct_total"] = 100.0 * cpu_used / wall_time
                summary["cpu_util_pct_per_core"] = 100.0 * cpu_used / (wall_time * c)
                summary["cpu_util_pct"] = summary["cpu_util_pct_total"]
            else:
                summary["cpu_util_pct_per_core"] = None
                summary["cpu_util_pct"] = None

        if self._gpu_samples:
            # Aggregate across GPUs
            all_gpu_utils = [s["gpu_util_pct"] for s in self._gpu_samples]
            all_mem_utils = [s["gpu_mem_util_pct"] for s in self._gpu_samples]
            all_mem_used = [s["gpu_mem_used_mb"] for s in self._gpu_samples]
            summary["gpu_util_pct"] = statistics.fmean(all_gpu_utils)
            summary["gpu_mem_util_pct"] = statistics.fmean(all_mem_utils)
            summary["gpu_mem_used_mb"] = max(all_mem_used)
            summary["gpu_mem_total_mb"] = max(s["gpu_mem_total_mb"] for s in self._gpu_samples)

            # Per-GPU aggregation
            gpu_ids_seen = set(s.get("gpu_id") for s in self._gpu_samples)
            for gid in sorted(gpu_ids_seen):
                gpu_samples = [s for s in self._gpu_samples if s.get("gpu_id") == gid]
                if gpu_samples:
                    per_gpu = {
                        "gpu_util_pct_avg": statistics.fmean(s["gpu_util_pct"] for s in gpu_samples),
                        "gpu_mem_used_mb_peak": max(s["gpu_mem_used_mb"] for s in gpu_samples),
                    }
                    if any("gpu_power_w" in s for s in gpu_samples):
                        power_vals = [s["gpu_power_w"] for s in gpu_samples if "gpu_power_w" in s]
                        if power_vals:
                            per_gpu["gpu_power_w_avg"] = statistics.fmean(power_vals)
                            per_gpu["gpu_power_w_peak"] = max(power_vals)
                    summary["per_gpu"][f"gpu_{gid}"] = per_gpu

        return summary


# ---------------------------------------------------------------------------
# Outlier exclusion
# ---------------------------------------------------------------------------
def exclude_outliers(values: list[float]) -> Tuple[list[float], int]:
    """Remove values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR] (Tukey fences)."""
    if len(values) < 4:
        return values, 0
    quantiles_out = statistics.quantiles(values, n=4)
    q1, q3 = quantiles_out[0], quantiles_out[2]
    iqr = q3 - q1
    if iqr == 0:
        return values, 0
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    filtered = [v for v in values if lower <= v <= upper]
    return filtered, len(values) - len(filtered)


# ---------------------------------------------------------------------------
# Statistics summarizer with CI
# ---------------------------------------------------------------------------
def _ci95(values: list[float], mean_val: float, std_val: float, *, smooth_window: int | None = None) -> Tuple[float, float]:
    """Compute 95% confidence interval on the mean, correcting for smoothing autocorrelation."""
    n = len(values)
    if n < 2:
        return math.nan, math.nan

    if smooth_window is not None and smooth_window > 1:
        eff = n / smooth_window
    else:
        eff = n

    if HAS_SCIPY:
        try:
            ci = scipy_stats.t.interval(0.95, df=max(int(round(eff)) - 1, 1), loc=mean_val, scale=std_val / (eff ** 0.5))
            return ci[0], ci[1]
        except Exception:
            pass

    # Manual approximation with effective sample count
    eff_int = int(round(eff))
    if eff_int <= 2:
        t_factor = 12.706
    elif eff_int <= 5:
        t_factor = 4.604
    elif eff_int <= 10:
        t_factor = 3.169
    elif eff_int <= 20:
        t_factor = 2.093
    elif eff_int <= 30:
        t_factor = 2.042
    else:
        t_factor = 1.960

    margin = t_factor * std_val / (eff ** 0.5)
    return mean_val - margin, mean_val + margin


def summarize(values: list[float], *, smooth_window: int | None = None) -> dict[str, float]:
    """Summarize a list of values with sample stdev, IQR outlier exclusion, and 95% CI."""
    if not values:
        return {
            "mean": math.nan, "std": math.nan, "median": math.nan,
            "min": math.nan, "max": math.nan,
            "ci95_low": math.nan, "ci95_high": math.nan,
            "outliers_removed": 0, "outlier_method": "IQR_Tukey_1.5",
            "ci95_note": None, "_cleaned": [],
        }

    # Exclude outliers via IQR method
    cleaned, n_removed = exclude_outliers(values)

    n = len(cleaned)
    mean_val = statistics.fmean(cleaned)
    std_val = statistics.stdev(cleaned) if n > 1 else 0.0
    median_val = statistics.median(cleaned)

    if n >= MIN_SAMPLES_FOR_CI:
        ci_low, ci_high = _ci95(cleaned, mean_val, std_val, smooth_window=smooth_window)
        if smooth_window is not None and smooth_window > 1 and n > smooth_window:
            eff_n = n / smooth_window
            ci_note = (
                f"CI95 computed with effective n={eff_n:.1f} (raw n={n}, smooth_window={smooth_window}). "
                "This corrects for autocorrelation introduced by moving-average smoothing."
            )
        else:
            ci_note = "CI95 assumes i.i.d. samples."
    else:
        ci_low, ci_high = math.nan, math.nan
        ci_note = (
            f"CI95 suppressed: only {n} data points (need >= {MIN_SAMPLES_FOR_CI}); "
            f"note: smoothed values (window=20) introduce auto-correlation. "
            f"Typically negative for this workload, making the CI conservative when computed."
        )

    # Floor CI95 bounds where negative values are impossible (time, throughput)
    if mean_val >= 0 and all(v >= 0 for v in cleaned):
        ci_low = max(ci_low, 0.0)

    return {
        "mean": round(mean_val, 6),
        "std": round(std_val, 6),
        "median": round(median_val, 6),
        "min": round(min(cleaned), 6),
        "max": round(max(cleaned), 6),
        "ci95_low": round(ci_low, 6) if not math.isnan(ci_low) else None,
        "ci95_high": round(ci_high, 6) if not math.isnan(ci_high) else None,
        "ci95_note": ci_note,
        "outliers_removed": n_removed,
        "outlier_method": "IQR_Tukey_1.5",
        "_cleaned": cleaned,
    }


# ---------------------------------------------------------------------------
# Benchmark metadata
# ---------------------------------------------------------------------------
def _get_driver_version() -> str:
    if HAS_PYNVML:
        try:
            pynvml.nvmlInit()
            ver = pynvml.nvmlSystemGetDriverVersion()
            pynvml.nvmlShutdown()
            return ver if isinstance(ver, str) else str(ver)
        except Exception:
            pass
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip().splitlines()[0].strip()
    except Exception:
        pass
    return "unknown"


def build_parser():
    parser = argparse.ArgumentParser(description="DINOv3 Production Telemetry Benchmark (v3.0)")
    parser.add_argument("--config-file", default="dinov3/configs/train/vitl_im1k_lin834.yaml", type=str)
    parser.add_argument("--output-dir", default="./output_dinov3_train_bench", type=str)
    parser.add_argument("--batch-size", default=None, type=int, help="Override local batch size per GPU")
    parser.add_argument("--num-workers", default=None, type=int, help="Override dataloader workers")
    parser.add_argument("--warmup-steps", default=5, type=int, help="Number of warmup steps to discard")
    parser.add_argument("--measure-steps", default=300, type=int, help="Number of steps to measure")
    parser.add_argument("--telemetry-period", default=0.25, type=float,
                        help="Telemetry sampling period in seconds (0.1 recommended with pynvml, 0.25 for subprocess)")
    parser.add_argument("--telemetry-backend", default=None, choices=["pynvml", "subprocess"],
                        help="Force telemetry backend (auto-detect by default)")
    parser.add_argument("--output-json", default="benchmark_results_train.json", type=str, help="Output summary file")
    parser.add_argument("--include-eval", action="store_true", help="Include evaluation overhead (disables eval_period_iterations=0 override)")
    parser.add_argument("--include-checkpoint", action="store_true", help="Include checkpointing overhead (disables checkpointing.period=99999 override)")
    return parser


def verify_dataset_and_dependencies():
    logger.info("Performing pre-run dataset and dependency verification...")
    try:
        import datasets
        import pyarrow.dataset
        from PIL import Image
    except ImportError as e:
        logger.error(f"Missing required dependency for pathology dataset: {e}")
        logger.error("Please run: uv pip install datasets pyarrow pillow")
        sys.exit(1)

    dataset_dir = Path("/data/nanopath_parquet")
    if not dataset_dir.exists():
        logger.error(f"Local pathology dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    parquet_files = list(dataset_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"No parquet files found in {dataset_dir}")
        sys.exit(1)

    logger.info(f"Found {len(parquet_files)} parquet shards in {dataset_dir}.")

    logger.info("Attempting to stream a single sample from the local parquet files...")
    try:
        from datasets import load_dataset
        import io

        data_files = [str(p) for p in parquet_files]
        dataset = load_dataset("parquet", data_files=data_files, split="train", streaming=True)

        sample = next(iter(dataset))
        img_bytes = sample.get("jpeg") or sample.get("image_bytes")
        if img_bytes is None:
            raise KeyError("Neither 'jpeg' nor 'image_bytes' key was found in the parquet schema.")

        img = Image.open(io.BytesIO(img_bytes))
        img.verify()
        logger.info(
            f"Pre-run dataset check PASSED! Successfully read sample of size {img.size} "
            f"from {Path(sample['path']).name if 'path' in sample else 'unknown path'} "
            f"without downloading anything."
        )
    except Exception as e:
        logger.error(f"Pre-run dataset verification FAILED: {e}")
        logger.error("Ensure that dataset dependencies are correctly installed and that the parquet files are not corrupted.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# JSON serialization helpers
# ---------------------------------------------------------------------------
def sanitize_for_json(obj):
    """Recursively convert NaN, +/-Inf to None for valid JSON serialization.
    Also strips internal '_'-prefixed keys from dicts and converts Path to str."""
    if isinstance(obj, dict):
        return {
            k: sanitize_for_json(v)
            for k, v in obj.items()
            if not k.startswith("_")
        }
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Benchmark sub-functions (decomposed from run_benchmark)
# ---------------------------------------------------------------------------
def _parse_args():
    """Parse and return CLI arguments."""
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    return args, unknown


def _setup_environment(args, unknown):
    """Create output directory and build fake_argv for training launch."""
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = args.warmup_steps + args.measure_steps

    fake_argv = [
        "train.py",
        "--config-file", args.config_file,
        "--output-dir", str(output_dir),
        "--no-resume",
        f"train.OFFICIAL_EPOCH_LENGTH={total_steps}",
        "optim.epochs=1",
        "optim.warmup_epochs=0",
        "optim.freeze_last_layer_epochs=0",
        "teacher.warmup_teacher_temp_epochs=0",
        "train.dataset_path=pathology:root=/data/nanopath_parquet",
    ]

    # Only suppress eval/checkpointing when NOT explicitly requested
    if not args.include_eval:
        fake_argv.append("evaluation.eval_period_iterations=0")
    if not args.include_checkpoint:
        fake_argv.append("checkpointing.period=99999")

    if args.batch_size is not None:
        fake_argv.append(f"train.batch_size_per_gpu={args.batch_size}")
    if args.num_workers is not None:
        fake_argv.append(f"train.num_workers={args.num_workers}")

    fake_argv.extend(unknown)
    return output_dir, total_steps, fake_argv


def _launch_training(args, output_dir, fake_argv, telemetry):
    """Launch training, handle exceptions (including SystemExit), stop telemetry."""
    period = args.telemetry_period
    telemetry_backend_actual = args.telemetry_backend or ("pynvml" if HAS_PYNVML else "subprocess")
    if telemetry_backend_actual == "subprocess" and period < 1.0:
        logger.warning(
            f"subprocess backend with period={period}s; nvidia-smi reports ~1s rolling averages, "
            f"effective sampling will be lower. Consider setting --telemetry-period={DEFAULT_TELEMETRY_PERIOD_SUBPROCESS} "
            f"or using pynvml backend."
        )

    benchmark_start = time.perf_counter()
    benchmark_end = benchmark_start
    original_argv = sys.argv
    training_failed = False
    training_tb = None
    try:
        from dinov3.train.train import main as train_main
        sys.argv = ["train.py", str(output_dir)]
        train_main(fake_argv)
        benchmark_end = time.perf_counter()
    except SystemExit as exc:
        benchmark_end = time.perf_counter()
        training_failed = True
        training_tb = f"Training exited via sys.exit({exc.code})"
        logger.error(f"Training exited via sys.exit({exc.code})")
    except Exception as exc:
        benchmark_end = time.perf_counter()
        training_failed = True
        training_tb = traceback.format_exc()
        logger.error(f"Training failed with exception: {exc}")
        logger.error(training_tb)
    finally:
        sys.argv = original_argv
        if telemetry is not None:
            telemetry.stop()
    return benchmark_start, benchmark_end, training_failed, training_tb


def _parse_metrics_file(output_dir):
    """Parse training_metrics.json line-by-line and return raw data arrays."""
    metrics_file = output_dir / "training_metrics.json"
    raw_iterations = []
    raw_step_times = []
    raw_data_times = []
    if metrics_file.exists():
        with open(metrics_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    raw_iterations.append(data.get("iteration", 0))
                    raw_step_times.append(data.get("iter_time", 0.0))
                    raw_data_times.append(data.get("data_time", 0.0))
                except Exception as e:
                    logger.warning(f"Error parsing metrics line: {e}")
    return raw_iterations, raw_step_times, raw_data_times


def _infer_and_filter_warmup(args, raw_iterations, raw_step_times, raw_data_times):
    """Infer print_freq, align warmup, skip transients, return filtered arrays."""
    inferred_print_freq = 1
    if len(raw_iterations) >= 2:
        diffs = [raw_iterations[i + 1] - raw_iterations[i] for i in range(len(raw_iterations) - 1)]
        diffs = [d for d in diffs if d > 0]
        if diffs:
            inferred_print_freq = int(statistics.median(diffs))

    # Align warmup to nearest print_freq boundary, then add one full cycle
        aligned_warmup = math.ceil(args.warmup_steps / inferred_print_freq) * inferred_print_freq
        effective_warmup = aligned_warmup
    if effective_warmup > args.warmup_steps:
        logger.warning(
            f"Warmup adjusted: requested warmup_steps={args.warmup_steps}, "
            f"but inferred print_freq={inferred_print_freq}; using effective_warmup={effective_warmup}"
        )

    step_index = None
    for i, it in enumerate(raw_iterations):
        if it >= effective_warmup:
            step_index = i
            break

    if step_index is not None:
        remaining = len(raw_iterations) - step_index
        skip = min(DEFAULT_SKIP_TRANSIENT, max(0, remaining - MIN_SAMPLES))
        step_index += skip

    if step_index is not None:
        step_times = raw_step_times[step_index:]
        data_times = raw_data_times[step_index:]
    else:
        step_times = []
        data_times = []

    return step_times, data_times, effective_warmup, inferred_print_freq, step_index


def _load_batch_sizes(args, output_dir):
    """Load batch sizes from config or fall back to args/defaults."""
    batch_size_per_gpu = args.batch_size or DEFAULT_BATCH_SIZE
    if args.batch_size is None:
        logger.warning(
            f"batch_size not specified, defaulting to {batch_size_per_gpu} — "
            f"throughput calculations may be inaccurate if config uses a different value"
        )
    global_batch_size = batch_size_per_gpu
    config_yaml_path = output_dir / "config.yaml"
    if config_yaml_path.exists():
        try:
            from omegaconf import OmegaConf
            cfg = OmegaConf.load(config_yaml_path)
            batch_size_per_gpu = cfg.train.batch_size_per_gpu
            from dinov3 import distributed
            world_size = distributed.get_world_size() if distributed.is_enabled() else 1
            global_batch_size = batch_size_per_gpu * world_size
        except Exception as e:
            logger.warning(f"Could not load written config for batch size details: {e}")
    return batch_size_per_gpu, global_batch_size


def _assemble_results(
    args, wall_time_s, training_failed, training_tb,
    step_times, data_times, status_val,
    batch_size_per_gpu, global_batch_size,
    effective_warmup, inferred_print_freq, step_index,
    raw_iterations, telemetry_summary,
    peak_gpu_mem_allocated_mb, peak_gpu_mem_reserved_mb,
):
    """Build the overall_results dict from all computed metrics."""
    SMOOTH_W = 20
    step_summary = summarize(step_times, smooth_window=SMOOTH_W)
    data_summary = summarize(data_times, smooth_window=SMOOTH_W)

    data_time_note = (
        "data_time from training_metrics.json measures the gap between Python loop "
        "iterations (helpers.py:log_every line 97: data_time.update(time.time() - end)), "
        "NOT actual data loading time."
    )

    # Compute throughput from cleaned step_times after outlier removal
    cleaned_step_times = step_summary.get("_cleaned", step_times)
    throughputs = [global_batch_size / step for step in cleaned_step_times if step > 0]
    throughput_summary = summarize(throughputs, smooth_window=SMOOTH_W)

    # cpu_gpu_ratio: data_time / step_time
    if (step_summary["mean"] and not math.isnan(step_summary["mean"]) and
            data_summary["mean"] and not math.isnan(data_summary["mean"])):
        cpu_gpu_ratio = round(data_summary["mean"] / step_summary["mean"], 4)
    else:
        cpu_gpu_ratio = None

    # compute_only_wall_time_s
    if step_summary["mean"] and not math.isnan(step_summary["mean"]):
        compute_only_wall_time_s = round(len(cleaned_step_times) * step_summary["mean"], 3)
    else:
        compute_only_wall_time_s = None

    # Effective warmup note with measurement window
    effective_start_iter = (raw_iterations[step_index]
                            if step_index is not None and step_index < len(raw_iterations)
                            else None)
    effective_end_iter = raw_iterations[-1] if raw_iterations else None
    warmup_note = (
        f"Warmup: requested warmup_steps={args.warmup_steps}, inferred print_freq="
        f"{inferred_print_freq}, effective_warmup={effective_warmup}, "
        f"transient_skip={DEFAULT_SKIP_TRANSIENT}, "
        f"measurement window: iterations {effective_start_iter}-{effective_end_iter} "
        f"({len(step_times)} logged samples)"
    )

    # GPU util imbalance
    per_gpu = telemetry_summary.get("per_gpu", {})
    if len(per_gpu) > 1:
        gpu_utils = [v["gpu_util_pct_avg"] for v in per_gpu.values() if "gpu_util_pct_avg" in v]
        gpu_util_imbalance_pp = round(max(gpu_utils) - min(gpu_utils), 2) if len(gpu_utils) > 1 else 0.0
    else:
        gpu_util_imbalance_pp = 0.0
    telemetry_summary["gpu_util_imbalance_pp"] = gpu_util_imbalance_pp

    # Build unrealistic overrides list based on what was actually applied
    unrealistic_overrides = ["optim.warmup_epochs=0", "teacher.warmup_teacher_temp_epochs=0"]
    if not args.include_eval:
        unrealistic_overrides.append("evaluation.eval_period_iterations=0")
    if not args.include_checkpoint:
        unrealistic_overrides.append("checkpointing.period=99999")

    driver_version = _get_driver_version()
    benchmark_metadata = {
        "benchmark_version": BENCHMARK_VERSION,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "N/A",
        "nvidia_driver_version": driver_version,
        "python_version": (f"{platform.python_implementation()} {platform.python_version()} "
                           f"({platform.system()} {platform.release()})"),
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        models = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        unique = list(dict.fromkeys(models))
        benchmark_metadata["gpu_model"] = "; ".join(unique)

    overall_results = {
        "status": status_val,
        "metadata": benchmark_metadata,
        "config_file": args.config_file,
        "batch_size_per_gpu": batch_size_per_gpu,
        "global_batch_size": global_batch_size,
        "warmup_steps": args.warmup_steps,
        "measure_steps": args.measure_steps,
        "total_wall_time_s": round(wall_time_s, 3),
        "compute_only_wall_time_s": compute_only_wall_time_s,
        "wall_time_includes_init": True,
        "step_time_s": step_summary,
        "data_time_s": data_summary,
        "data_time_note": data_time_note,
        "throughput_images_per_s": throughput_summary,
        "cpu_gpu_ratio": cpu_gpu_ratio,
        "cpu_gpu_ratio_note": "data_time / step_time — fraction of each step spent in non-compute (data load overhead, loop scheduling)",
        "peak_gpu_mem_allocated_mb": round(peak_gpu_mem_allocated_mb, 2) if peak_gpu_mem_allocated_mb is not None else None,
        "peak_gpu_mem_reserved_mb": round(peak_gpu_mem_reserved_mb, 2) if peak_gpu_mem_reserved_mb is not None else None,
        "telemetry": telemetry_summary,
        "metrics_are_smoothed": True,
        "smooth_window": SMOOTH_W,
        "actual_data_points": len(step_times),
        "warmup_note": warmup_note,
        "unrealistic_config_overrides": unrealistic_overrides,
    }

    if training_failed and training_tb:
        overall_results["traceback"] = training_tb

    return overall_results, step_summary, data_summary, throughput_summary, data_time_note


def _print_summary(
    args, overall_results, telemetry_summary, status_val,
    step_summary, data_summary, throughput_summary, data_time_note,
    benchmark_metadata, peak_gpu_mem_allocated_mb, peak_gpu_mem_reserved_mb,
):
    """Print formatted benchmark results to logger."""
    batch_size_per_gpu = overall_results["batch_size_per_gpu"]
    global_batch_size = overall_results["global_batch_size"]

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"         DINOv3 BENCHMARK RESULTS SUMMARY (v{BENCHMARK_VERSION})")
    logger.info("=" * 60)
    logger.info(f"Model Configuration:      {args.config_file}")
    logger.info(f"Batch Size (Local/Global): {batch_size_per_gpu} / {global_batch_size}")
    if status_val != "OK":
        logger.error(f"Status: {status_val} — metrics below may be NaN or incomplete")
        logger.info(f"Compute wall time:    {overall_results['compute_only_wall_time_s']} s (estimate)")
    else:
        ci_str = ""
        if step_summary["ci95_low"] is not None:
            ci_str = f", 95% CI [{step_summary['ci95_low']*1000:.2f}, {step_summary['ci95_high']*1000:.2f}]"
        else:
            ci_str = " [CI suppressed, insufficient samples]"
        logger.info(
            f"Step Time:        {step_summary['mean']*1000:.2f} ms +/- {step_summary['std']*1000:.2f} ms "
            f"(median {step_summary['median']*1000:.2f} ms{ci_str})"
        )
        logger.info("  [20-window smoothed, std/CI approximate]")
        if step_summary["outliers_removed"] > 0:
            logger.info(f"  [outliers excluded: {step_summary['outliers_removed']}]")

        logger.info(
            f"Throughput:        {throughput_summary['mean']:.2f} images/s +/- {throughput_summary['std']:.2f}"
        )

        logger.info(
            f"CPU/GPU ratio:     {overall_results['cpu_gpu_ratio']}"
        )

        logger.info(
            f"Data Fetch time:   {data_summary['mean']*1000:.2f} ms +/- {data_summary['std']*1000:.2f} ms"
        )
        logger.warning(f"DATA_TIME NOTE: {data_time_note}")

        compute_wall = overall_results.get("compute_only_wall_time_s")
        logger.info(f"Compute wall time:  {compute_wall} s (estimate from cleaned samples)")

    if telemetry_summary["gpu_util_pct"] is not None:
        logger.info(f"Avg GPU Util:         {telemetry_summary['gpu_util_pct']:.1f}%")
        logger.info(
            f"Peak GPU Mem (nvidia-smi): {telemetry_summary['gpu_mem_used_mb']:.0f} / "
            f"{telemetry_summary['gpu_mem_total_mb']:.0f} MB"
        )
    if peak_gpu_mem_allocated_mb is not None:
        alloc_str = f"{peak_gpu_mem_allocated_mb:.0f} MB allocated"
        if peak_gpu_mem_reserved_mb is not None:
            alloc_str += f", {peak_gpu_mem_reserved_mb:.0f} MB reserved"
        logger.info(f"Peak GPU Mem (torch):   {alloc_str}")

    if telemetry_summary.get("per_gpu"):
        for gid, per in telemetry_summary["per_gpu"].items():
            logger.info(f"  {gid}: util={per['gpu_util_pct_avg']:.1f}%, "
                        f"mem_peak={per['gpu_mem_used_mb_peak']:.0f} MB")

    imbalance = telemetry_summary.get("gpu_util_imbalance_pp")
    if imbalance is not None and imbalance > 0:
        logger.info(f"GPU util imbalance:   {imbalance:.1f} pp (max-min across GPUs)")

    if telemetry_summary["cpu_util_pct"] is not None:
        logger.info(f"Avg CPU Util:         {telemetry_summary['cpu_util_pct']:.1f}%")
        logger.info(f"Peak CPU RSS:         {telemetry_summary['rss_mb']:.2f} MB")

    logger.info("=" * 60)
    logger.info(f"Bench version:      {benchmark_metadata['benchmark_version']}")
    logger.info(f"PyTorch:            {benchmark_metadata['pytorch_version']}")
    logger.info(f"CUDA:               {benchmark_metadata['cuda_version']}")
    logger.info(f"NV driver:          {benchmark_metadata['nvidia_driver_version']}")
    if benchmark_metadata.get("num_gpus", 0) > 0:
        logger.info(f"GPU model:          {benchmark_metadata.get('gpu_model', 'N/A')}")
    logger.info(f"Telemetry backend:  {telemetry_summary.get('backend', 'N/A')}")
    logger.info("=" * 60)


def _write_results(overall_results, output_json_path):
    """Atomically write benchmark results to JSON with backup of existing file."""
    if output_json_path.exists():
        try:
            mtime = output_json_path.stat().st_mtime
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(mtime))
            backup_path = output_json_path.with_name(f"{output_json_path.stem}_{timestamp}{output_json_path.suffix}")
            output_json_path.rename(backup_path)
            logger.info(f"Backed up existing benchmark results to {backup_path}")
        except Exception as e:
            logger.warning(f"Failed to backup existing benchmark results file: {e}")

    tmp_path = output_json_path.with_suffix(output_json_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(sanitize_for_json(overall_results), f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp_path), str(output_json_path))
    logger.info(f"Saved benchmark summary to {output_json_path}")


def run_benchmark():
    """Main entry point — delegates to helper functions."""
    args, unknown = _parse_args()
    is_main = os.environ.get("RANK", "0") == "0"

    if is_main:
        verify_dataset_and_dependencies()

    output_dir, total_steps, fake_argv = _setup_environment(args, unknown)

    telemetry = None
    if is_main:
        logger.info(f"Launching production training benchmark for {total_steps} steps...")
        logger.info(f"Warmup iterations: {args.warmup_steps}")
        logger.info(f"Measured iterations: {args.measure_steps}")

        period = args.telemetry_period
        telemetry = TelemetrySampler(
            sample_period_s=period,
            telemetry_backend=args.telemetry_backend,
        )
        telemetry.start()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    benchmark_start, benchmark_end, training_failed, training_tb = (
        _launch_training(args, output_dir, fake_argv, telemetry)
    )

    if not is_main:
        return

    wall_time_s = benchmark_end - benchmark_start
    telemetry_summary = telemetry.summary(benchmark_start, benchmark_end)

    peak_gpu_mem_allocated_mb = None
    peak_gpu_mem_reserved_mb = None
    if torch.cuda.is_available() and telemetry_summary.get("gpu_mem_used_mb"):
        peak_alloc_torch = torch.cuda.max_memory_allocated() / (1024.0**2)
        peak_gpu_mem_allocated_mb = max(telemetry_summary["gpu_mem_used_mb"], peak_alloc_torch)
        peak_gpu_mem_reserved_mb = torch.cuda.max_memory_reserved() / (1024.0**2)
        if torch.cuda.device_count() > 1:
            logger.info(f"Peak GPU memory (telemetry max across GPUs): {peak_gpu_mem_allocated_mb:.1f} MB")
            logger.info(f"Peak GPU memory (rank-0 torch allocated): {peak_alloc_torch:.1f} MB")
        else:
            logger.info(f"Peak GPU memory allocated: {peak_gpu_mem_allocated_mb:.1f} MB")

    raw_iterations, raw_step_times, raw_data_times = _parse_metrics_file(output_dir)
    step_times, data_times, effective_warmup, inferred_print_freq, step_index = (
        _infer_and_filter_warmup(args, raw_iterations, raw_step_times, raw_data_times)
    )
    batch_size_per_gpu, global_batch_size = _load_batch_sizes(args, output_dir)

    # Validate minimum sample size
    if len(step_times) == 0:
        logger.error("No step data collected — training may have failed")
        status_val = "FAILED" if training_failed else "ERROR_NO_DATA"
    elif training_failed:
        status_val = "FAILED"
    elif len(step_times) < MIN_SAMPLES:
        logger.error(
            f"Insufficient data points: {len(step_times)} (need >= {MIN_SAMPLES}). "
            f"Increase --measure-steps (current {args.measure_steps}) so that "
            f"measure_steps / print_freq({inferred_print_freq}) >= {MIN_SAMPLES}. "
            f"Suggested: --measure-steps={MIN_SAMPLES * inferred_print_freq}"
        )
        status_val = "ERROR_INSUFFICIENT_DATA"
    else:
        status_val = "OK"

    if 10 <= len(step_times) < 20:
        logger.warning(
            f"Low sample size: {len(step_times)} data points "
            f"(recommended >= 20, expected ~{args.measure_steps})"
        )

    (overall_results, step_summary, data_summary, throughput_summary,
     data_time_note) = _assemble_results(
        args, wall_time_s, training_failed, training_tb,
        step_times, data_times, status_val,
        batch_size_per_gpu, global_batch_size,
        effective_warmup, inferred_print_freq, step_index,
        raw_iterations, telemetry_summary,
        peak_gpu_mem_allocated_mb, peak_gpu_mem_reserved_mb,
    )

    # Check for GPU imbalance after _assemble_results (which computes gpu_util_imbalance_pp)
    if status_val == "OK" and telemetry_summary.get("gpu_util_imbalance_pp") is not None:
        imbalance = telemetry_summary["gpu_util_imbalance_pp"]
        if imbalance > 30.0:
            status_val = "DEGRADED"
            overall_results["status"] = "DEGRADED"
            logger.warning(
                f"GPU utilization imbalance too high ({imbalance:.1f} pp) — status set to DEGRADED. "
                f"Throughput may be limited by the slowest GPU."
            )

    _print_summary(
        args, overall_results, telemetry_summary, status_val,
        step_summary, data_summary, throughput_summary, data_time_note,
        overall_results["metadata"], peak_gpu_mem_allocated_mb, peak_gpu_mem_reserved_mb,
    )

    _write_results(overall_results, Path(args.output_json))

    if status_val in ("FAILED", "ERROR_NO_DATA", "ERROR_INSUFFICIENT_DATA"):
        sys.exit(1)


if __name__ == "__main__":
    run_benchmark()
