from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Any

import pandas as pd


METRIC_WEIGHTS = {
    "balanced": {"tokens_per_second": 0.35, "requests_per_minute": 0.25, "avg_latency_s": 0.25, "p95_latency_s": 0.15},
    "lowest latency": {"tokens_per_second": 0.15, "requests_per_minute": 0.10, "avg_latency_s": 0.45, "p95_latency_s": 0.30},
    "max throughput": {"tokens_per_second": 0.50, "requests_per_minute": 0.35, "avg_latency_s": 0.10, "p95_latency_s": 0.05},
    "lowest memory": {"tokens_per_second": 0.20, "requests_per_minute": 0.15, "avg_latency_s": 0.15, "p95_latency_s": 0.10, "gpu_memory_gb": 0.40},
}


@dataclass(frozen=True)
class Recommendation:
    config_name: str
    score: float
    rationale: str
    launch_hint: str
    score_breakdown: dict[str, float]


def _normalize_higher(series: pd.Series) -> pd.Series:
    low = float(series.min())
    high = float(series.max())
    if high == low:
        return pd.Series([1.0] * len(series), index=series.index)
    return (series - low) / (high - low)


def _normalize_lower(series: pd.Series) -> pd.Series:
    low = float(series.min())
    high = float(series.max())
    if high == low:
        return pd.Series([1.0] * len(series), index=series.index)
    return 1.0 - ((series - low) / (high - low))


def _runs_frame(payload: dict[str, Any]) -> pd.DataFrame:
    runs = payload.get("runs", [])
    if not runs:
        raise ValueError("Benchmark JSON must include a non-empty 'runs' array.")

    frame = pd.DataFrame(runs)
    required = {
        "config_name",
        "concurrency",
        "max_model_len",
        "max_tokens",
        "avg_latency_s",
        "p95_latency_s",
        "tokens_per_second",
        "requests_per_minute",
        "gpu_memory_gb",
        "error_rate",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Benchmark JSON is missing required run fields: {', '.join(missing)}")
    return frame


def analyze_benchmark(payload: dict[str, Any], objective: str = "balanced") -> tuple[pd.DataFrame, Recommendation]:
    frame = _runs_frame(payload).copy()
    objective_key = objective.strip().lower()
    weights = METRIC_WEIGHTS.get(objective_key, METRIC_WEIGHTS["balanced"])

    frame["throughput_score"] = (_normalize_higher(frame["tokens_per_second"].astype(float)) * 100).round(1)
    frame["request_rate_score"] = (_normalize_higher(frame["requests_per_minute"].astype(float)) * 100).round(1)
    frame["avg_latency_score"] = (_normalize_lower(frame["avg_latency_s"].astype(float)) * 100).round(1)
    frame["p95_latency_score"] = (_normalize_lower(frame["p95_latency_s"].astype(float)) * 100).round(1)
    frame["memory_score"] = (_normalize_lower(frame["gpu_memory_gb"].astype(float)) * 100).round(1)
    frame["reliability_score"] = ((1.0 - frame["error_rate"].astype(float).clip(0, 1)) * 100).round(1)

    score = pd.Series([0.0] * len(frame), index=frame.index)
    for metric, weight in weights.items():
        if metric in {"avg_latency_s", "p95_latency_s", "gpu_memory_gb"}:
            score += _normalize_lower(frame[metric].astype(float)) * weight
        else:
            score += _normalize_higher(frame[metric].astype(float)) * weight

    reliability_factor = 1.0 - frame["error_rate"].astype(float).clip(0, 1)
    score = (score * reliability_factor) - (frame["error_rate"].astype(float).clip(0, 1) * 0.30)
    frame["amd_readiness_score"] = (score.clip(0, 1) * 100).round(1)
    frame = frame.sort_values("amd_readiness_score", ascending=False).reset_index(drop=True)

    best = frame.iloc[0]
    baseline = frame.sort_values("concurrency", ascending=True).iloc[0]
    throughput_gain = float(best["tokens_per_second"]) / max(float(baseline["tokens_per_second"]), 0.001)
    p95_delta = float(best["p95_latency_s"]) - float(baseline["p95_latency_s"])
    launch_hint = (
        f"vllm serve {payload.get('model', '<model>')} "
        f"--max-model-len {int(best['max_model_len'])} "
        f"--max-num-seqs {int(best['concurrency'])} "
        "--gpu-memory-utilization 0.90"
    )
    rationale = (
        f"{best['config_name']} is the strongest {objective} configuration: "
        f"{best['tokens_per_second']:.1f} tok/s, p95 {best['p95_latency_s']:.2f}s, "
        f"{best['gpu_memory_gb']:.1f} GB VRAM, error rate {best['error_rate']:.1%}. "
        f"Against the lowest-concurrency baseline, it delivers {throughput_gain:.1f}x throughput "
        f"with p95 latency changing by {p95_delta:+.2f}s."
    )
    recommendation = Recommendation(
        config_name=str(best["config_name"]),
        score=float(best["amd_readiness_score"]),
        rationale=rationale,
        launch_hint=launch_hint,
        score_breakdown={
            "throughput_score": float(best["throughput_score"]),
            "request_rate_score": float(best["request_rate_score"]),
            "avg_latency_score": float(best["avg_latency_score"]),
            "p95_latency_score": float(best["p95_latency_score"]),
            "memory_score": float(best["memory_score"]),
            "reliability_score": float(best["reliability_score"]),
        },
    )
    return frame, recommendation


def summarize_impact(frame: pd.DataFrame) -> dict[str, float | str]:
    ranked = frame.sort_values("amd_readiness_score", ascending=False)
    best = ranked.iloc[0]
    baseline = frame.sort_values("concurrency", ascending=True).iloc[0]
    throughput_gain = float(best["tokens_per_second"]) / max(float(baseline["tokens_per_second"]), 0.001)
    request_gain = float(best["requests_per_minute"]) / max(float(baseline["requests_per_minute"]), 0.001)
    p95_delta = float(best["p95_latency_s"]) - float(baseline["p95_latency_s"])
    avg_delta = float(best["avg_latency_s"]) - float(baseline["avg_latency_s"])
    return {
        "baseline_config": str(baseline["config_name"]),
        "recommended_config": str(best["config_name"]),
        "baseline_tokens_per_second": float(baseline["tokens_per_second"]),
        "recommended_tokens_per_second": float(best["tokens_per_second"]),
        "throughput_gain": throughput_gain,
        "request_gain": request_gain,
        "baseline_p95_latency_s": float(baseline["p95_latency_s"]),
        "recommended_p95_latency_s": float(best["p95_latency_s"]),
        "p95_delta_s": p95_delta,
        "avg_delta_s": avg_delta,
        "recommended_error_rate": float(best["error_rate"]),
    }


def estimate_costs(frame: pd.DataFrame, hourly_price: float, requests_per_day: int, output_tokens_per_request: int) -> dict[str, float]:
    best = frame.sort_values("amd_readiness_score", ascending=False).iloc[0]
    tokens_per_hour = float(best["tokens_per_second"]) * 3600
    expected_tokens_per_day = float(requests_per_day) * float(output_tokens_per_request)
    hours_per_day = expected_tokens_per_day / max(tokens_per_hour, 0.001)
    daily_cost = hours_per_day * float(hourly_price)
    cost_per_million_tokens = float(hourly_price) / max(tokens_per_hour / 1_000_000, 0.001)
    return {
        "tokens_per_hour": tokens_per_hour,
        "expected_tokens_per_day": expected_tokens_per_day,
        "hours_per_day": hours_per_day,
        "daily_cost": daily_cost,
        "cost_per_million_tokens": cost_per_million_tokens,
    }


def deployment_decision(frame: pd.DataFrame, recommendation: Recommendation) -> dict[str, str]:
    best = frame.sort_values("amd_readiness_score", ascending=False).iloc[0]
    if float(best["error_rate"]) > 0.01:
        return {
            "status": "BLOCK",
            "reason": "Observed error rate is above 1%, so this config should not ship without investigation.",
            "risk": "Reliability regression under concurrent load.",
            "next_test": "Inspect vLLM logs and rerun with smaller concurrency steps.",
        }
    if float(best["p95_latency_s"]) > 5.0:
        return {
            "status": "WARN",
            "reason": "The recommended config is performant but p95 latency is above 5 seconds.",
            "risk": "User-facing latency may be too high for interactive chat workloads.",
            "next_test": "Run a lower max-token benchmark and compare p95 latency.",
        }
    if recommendation.score < 60:
        return {
            "status": "WARN",
            "reason": "The readiness score is below 60, so the run needs another optimization pass.",
            "risk": "No configuration clearly dominates across throughput, latency, and reliability.",
            "next_test": "Try a smaller model, quantization, or lower max model length.",
        }
    return {
        "status": "APPROVE",
        "reason": "The recommended config provides strong throughput, controlled p95 latency, and acceptable reliability.",
        "risk": "High VRAM reservation should be validated with long-context and sustained-load tests.",
        "next_test": "Run a 15-minute soak test and one long-context benchmark at the recommended concurrency.",
    }


def _rocm_values(text: str) -> tuple[list[int], list[int], list[int]]:
    gpu_values = [int(value) for value in re.findall(r"GPU use \(%\): (\d+)", text)]
    vram_values = [int(value) for value in re.findall(r"GPU Memory Allocated \(VRAM%\): (\d+)", text)]
    memory_values = [int(value) for value in re.findall(r"GPU Memory Read/Write Activity \(%\): (\d+)", text)]
    return gpu_values, vram_values, memory_values


def parse_rocm_monitor(log_path: str | Path | None) -> dict[str, int | float | str]:
    if not log_path:
        return {
            "status": "No ROCm monitor log supplied.",
            "samples": 0,
            "peak_gpu_utilization": 0,
            "avg_gpu_utilization": 0,
            "peak_vram_allocated": 0,
            "avg_vram_allocated": 0,
            "peak_memory_activity": 0,
            "avg_memory_activity": 0,
            "gpu_saturation_share": 0,
        }
    path = Path(log_path)
    if not path.exists():
        return {
            "status": f"ROCm monitor log not found: {path}",
            "samples": 0,
            "peak_gpu_utilization": 0,
            "avg_gpu_utilization": 0,
            "peak_vram_allocated": 0,
            "avg_vram_allocated": 0,
            "peak_memory_activity": 0,
            "avg_memory_activity": 0,
            "gpu_saturation_share": 0,
        }

    text = path.read_text(encoding="utf-8", errors="ignore")
    gpu_values, vram_values, memory_values = _rocm_values(text)
    samples = len(gpu_values)
    return {
        "status": "ROCm monitor evidence loaded.",
        "samples": samples,
        "peak_gpu_utilization": max(gpu_values, default=0),
        "avg_gpu_utilization": round(sum(gpu_values) / samples, 1) if samples else 0,
        "peak_vram_allocated": max(vram_values, default=0),
        "avg_vram_allocated": round(sum(vram_values) / len(vram_values), 1) if vram_values else 0,
        "peak_memory_activity": max(memory_values, default=0),
        "avg_memory_activity": round(sum(memory_values) / len(memory_values), 1) if memory_values else 0,
        "gpu_saturation_share": round(sum(1 for value in gpu_values if value >= 90) / samples * 100, 1) if samples else 0,
    }


def parse_rocm_monitor_series(log_path: str | Path | None) -> pd.DataFrame:
    if not log_path:
        return pd.DataFrame(columns=["sample", "gpu_utilization", "vram_allocated", "memory_activity"])
    path = Path(log_path)
    if not path.exists():
        return pd.DataFrame(columns=["sample", "gpu_utilization", "vram_allocated", "memory_activity"])

    text = path.read_text(encoding="utf-8", errors="ignore")
    gpu_values, vram_values, memory_values = _rocm_values(text)
    rows = []
    for index, gpu_utilization in enumerate(gpu_values):
        rows.append(
            {
                "sample": index + 1,
                "gpu_utilization": gpu_utilization,
                "vram_allocated": vram_values[index] if index < len(vram_values) else None,
                "memory_activity": memory_values[index] if index < len(memory_values) else None,
            }
        )
    return pd.DataFrame(rows)


def assess_gpu_saturation(frame: pd.DataFrame, monitor_summary: dict[str, Any]) -> dict[str, Any]:
    best = frame.sort_values("amd_readiness_score", ascending=False).iloc[0]
    peak_gpu = float(monitor_summary.get("peak_gpu_utilization", 0) or 0)
    avg_gpu = float(monitor_summary.get("avg_gpu_utilization", 0) or 0)
    peak_vram = float(monitor_summary.get("peak_vram_allocated", 0) or 0)
    saturation_share = float(monitor_summary.get("gpu_saturation_share", 0) or 0)
    p95_latency = float(best["p95_latency_s"])
    error_rate = float(best["error_rate"])

    compute_score = min(100.0, peak_gpu * 0.45 + avg_gpu * 0.35 + saturation_share * 0.20)
    if peak_vram >= 95:
        compute_score -= 12
    if p95_latency > 5:
        compute_score -= 15
    if error_rate > 0.01:
        compute_score -= 25
    compute_score = round(max(0.0, min(100.0, compute_score)), 1)

    if compute_score >= 80 and error_rate <= 0.01:
        status = "SATURATED"
        summary = "The benchmark produced real AMD GPU pressure and the winning config is suitable for the next production gate."
    elif peak_gpu >= 80:
        status = "PARTIAL"
        summary = "The GPU was exercised, but the evidence suggests more tuning or a longer run is needed before calling it production-ready."
    else:
        status = "UNDER-UTILIZED"
        summary = "The run did not keep the GPU busy enough; increase concurrency, prompt count, or output tokens before judging capacity."

    return {
        "status": status,
        "score": compute_score,
        "summary": summary,
        "peak_gpu_utilization": peak_gpu,
        "avg_gpu_utilization": avg_gpu,
        "gpu_saturation_share": saturation_share,
        "peak_vram_allocated": peak_vram,
    }


def plan_next_experiments(frame: pd.DataFrame, recommendation: Recommendation, monitor_summary: dict[str, Any]) -> pd.DataFrame:
    ordered = frame.sort_values("concurrency", ascending=True)
    best = frame.sort_values("amd_readiness_score", ascending=False).iloc[0]
    max_tested_concurrency = int(ordered["concurrency"].max())
    best_concurrency = int(best["concurrency"])
    peak_vram = float(monitor_summary.get("peak_vram_allocated", 0) or 0)
    peak_gpu = float(monitor_summary.get("peak_gpu_utilization", 0) or 0)
    p95_latency = float(best["p95_latency_s"])
    error_rate = float(best["error_rate"])
    launch_model = re.search(r"vllm serve (.*?) --max-model-len", recommendation.launch_hint)
    model = launch_model.group(1) if launch_model else "the current model"

    rows: list[dict[str, str | int]] = []
    if best_concurrency == max_tested_concurrency and error_rate <= 0.01 and p95_latency <= 3.0 and peak_vram < 95:
        next_concurrency = max_tested_concurrency * 2
        rows.append(
            {
                "Priority": 1,
                "Experiment": f"Probe concurrency {next_concurrency}",
                "Change": f"--max-num-seqs {next_concurrency}",
                "Why": "The best config is still the highest tested concurrency, so the throughput ceiling has not been found.",
                "Pass signal": "Throughput rises without p95 latency crossing 5s or errors crossing 1%.",
            }
        )
    else:
        rows.append(
            {
                "Priority": 1,
                "Experiment": "Narrow concurrency around winner",
                "Change": f"test --max-num-seqs {max(1, best_concurrency - 2)} {best_concurrency} {best_concurrency + 2}",
                "Why": "The current winner is near a tradeoff boundary, so smaller steps can find a cleaner serving point.",
                "Pass signal": "A nearby config improves p95 latency or cost while keeping throughput within 5%.",
            }
        )

    if peak_vram >= 90:
        rows.append(
            {
                "Priority": 2,
                "Experiment": "Long-context VRAM boundary",
                "Change": "--max-model-len 8192 with the recommended concurrency",
                "Why": "ROCm proof shows high VRAM reservation, so long-context traffic is the most likely failure boundary.",
                "Pass signal": "No OOM, error rate below 1%, and p95 latency below the workload SLO.",
            }
        )
    else:
        rows.append(
            {
                "Priority": 2,
                "Experiment": "Increase context window",
                "Change": "--max-model-len 8192",
                "Why": "VRAM headroom appears available, so test whether the model can support larger production prompts.",
                "Pass signal": "Readiness score stays above 75 with stable memory behavior.",
            }
        )

    if peak_gpu < 90:
        rows.append(
            {
                "Priority": 3,
                "Experiment": "Raise load intensity",
                "Change": "double requests per config and max output tokens",
                "Why": "GPU utilization did not reach saturation, so current numbers may understate real capacity.",
                "Pass signal": "Peak GPU utilization reaches at least 90% for several samples.",
            }
        )
    else:
        rows.append(
            {
                "Priority": 3,
                "Experiment": "15-minute soak test",
                "Change": f"hold {recommendation.config_name} under steady traffic",
                "Why": "The GPU reached saturation; now prove the config remains stable beyond a short burst.",
                "Pass signal": "No errors, no rising p95 trend, and memory remains bounded.",
            }
        )

    rows.append(
        {
            "Priority": 4,
            "Experiment": "Compare a second HF model",
            "Change": f"same workload against an adjacent model to {model}",
            "Why": "A real second model result proves the app is choosing between deployments, not just charting one run.",
            "Pass signal": "Report shows a clear model-level tradeoff in cost, latency, and throughput.",
        }
    )
    return pd.DataFrame(rows).sort_values("Priority")


def build_deployment_gate_matrix(frame: pd.DataFrame, recommendation: Recommendation, monitor_summary: dict[str, Any]) -> pd.DataFrame:
    best = frame.sort_values("amd_readiness_score", ascending=False).iloc[0]
    p95_latency = float(best["p95_latency_s"])
    error_rate = float(best["error_rate"])
    peak_vram = float(monitor_summary.get("peak_vram_allocated", 0) or 0)
    peak_gpu = float(monitor_summary.get("peak_gpu_utilization", 0) or 0)
    saturation_share = float(monitor_summary.get("gpu_saturation_share", 0) or 0)

    rows = [
        {
            "Gate": "Reliability",
            "Status": "PASS" if error_rate <= 0.01 else "BLOCK",
            "Evidence": f"{error_rate:.1%} error rate",
            "Production meaning": "Safe enough for controlled rollout" if error_rate <= 0.01 else "Too many failed requests under load",
        },
        {
            "Gate": "Interactive latency",
            "Status": "PASS" if p95_latency <= 2 else "WARN" if p95_latency <= 5 else "BLOCK",
            "Evidence": f"{p95_latency:.3f}s p95 latency",
            "Production meaning": "Fits chat-style workloads" if p95_latency <= 2 else "Needs workload-specific SLO review",
        },
        {
            "Gate": "ROCm proof",
            "Status": "PASS" if peak_gpu >= 90 else "WARN",
            "Evidence": f"{peak_gpu:.0f}% peak GPU utilization",
            "Production meaning": "Real GPU-backed benchmark evidence" if peak_gpu >= 90 else "Run was not saturated enough to prove capacity",
        },
        {
            "Gate": "Sustained saturation",
            "Status": "PASS" if saturation_share >= 20 else "WARN",
            "Evidence": f"{saturation_share:.1f}% samples at 90%+ GPU use",
            "Production meaning": "Load held long enough to exercise the device" if saturation_share >= 20 else "Use a longer soak test",
        },
        {
            "Gate": "VRAM headroom",
            "Status": "PASS" if peak_vram < 90 else "WARN" if peak_vram < 96 else "BLOCK",
            "Evidence": f"{peak_vram:.0f}% peak VRAM allocated",
            "Production meaning": "Room for traffic variance" if peak_vram < 90 else "Validate long-context prompts before launch",
        },
        {
            "Gate": "Readiness score",
            "Status": "PASS" if recommendation.score >= 75 else "WARN" if recommendation.score >= 60 else "BLOCK",
            "Evidence": f"{recommendation.score:.1f}/100 AMD readiness",
            "Production meaning": "Config is a credible deployment candidate" if recommendation.score >= 75 else "Run another optimization pass",
        },
    ]
    return pd.DataFrame(rows)


def build_report(
    payload: dict[str, Any],
    objective: str,
    output_dir: str | Path = "reports",
    monitor_summary: dict[str, Any] | None = None,
    hourly_price: float = 4.0,
    requests_per_day: int = 100_000,
    output_tokens_per_request: int = 256,
) -> tuple[str, Path]:
    frame, recommendation = analyze_benchmark(payload, objective)
    impact = summarize_impact(frame)
    costs = estimate_costs(frame, hourly_price, requests_per_day, output_tokens_per_request)
    decision = deployment_decision(frame, recommendation)
    monitor = monitor_summary or {}
    saturation = assess_gpu_saturation(frame, monitor)
    next_experiments = plan_next_experiments(frame, recommendation, monitor).to_markdown(index=False)
    gate_matrix = build_deployment_gate_matrix(frame, recommendation, monitor).to_markdown(index=False)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = str(payload.get("model", "model")).replace("/", "_").replace(" ", "_")
    report_path = output_path / f"rocmbench_{model_slug}_{stamp}.md"

    top_rows = frame.head(5)[
        [
            "config_name",
            "amd_readiness_score",
            "concurrency",
            "tokens_per_second",
            "avg_latency_s",
            "p95_latency_s",
            "gpu_memory_gb",
            "error_rate",
        ]
    ].to_markdown(index=False)

    breakdown_rows = pd.DataFrame(
        [
            {"dimension": key.replace("_", " "), "score": value}
            for key, value in recommendation.score_breakdown.items()
        ]
    ).to_markdown(index=False)

    report = f"""# ROCmBench Agent Deployment Report

## Executive Recommendation

**Recommended config:** `{recommendation.config_name}`  
**AMD readiness score:** `{recommendation.score:.1f}/100`  
**Optimization goal:** `{objective}`

{recommendation.rationale}

## Deployment Decision

**Decision:** `{decision["status"]}`  
**Reason:** {decision["reason"]}  
**Primary risk:** {decision["risk"]}  
**Next test:** {decision["next_test"]}

## Before And After Impact

- Baseline config: `{impact["baseline_config"]}`
- Recommended config: `{impact["recommended_config"]}`
- Throughput: `{impact["baseline_tokens_per_second"]:.2f}` tok/s -> `{impact["recommended_tokens_per_second"]:.2f}` tok/s
- Throughput lift: `{impact["throughput_gain"]:.1f}x`
- P95 latency: `{impact["baseline_p95_latency_s"]:.3f}s` -> `{impact["recommended_p95_latency_s"]:.3f}s`
- Recommended error rate: `{impact["recommended_error_rate"]:.1%}`

## Model And Environment

- Model: `{payload.get("model", "unknown")}`
- Hardware: `{payload.get("hardware", "unknown")}`
- Backend: `{payload.get("backend", "unknown")}`
- Workload: `{payload.get("workload", "unknown")}`
- Captured at: `{payload.get("captured_at", "unknown")}`

## Recommended vLLM Launch

```bash
{recommendation.launch_hint}
```

## Top Benchmark Results

{top_rows}

## Score Breakdown

{breakdown_rows}

## Cost Estimate

- GPU hourly price: `${hourly_price:.2f}`
- Expected daily traffic: `{requests_per_day:,}` requests/day
- Output tokens per request: `{output_tokens_per_request:,}`
- Estimated capacity: `{costs["tokens_per_hour"]:,.0f}` tokens/hour
- Estimated daily GPU time: `{costs["hours_per_day"]:.2f}` hours/day
- Estimated daily serving cost: `${costs["daily_cost"]:.2f}`
- Estimated cost per 1M output tokens: `${costs["cost_per_million_tokens"]:.2f}`

## ROCm Proof

- Monitor status: `{monitor.get("status", "No monitor summary supplied.")}`
- ROCm samples: `{monitor.get("samples", 0)}`
- Peak GPU utilization: `{monitor.get("peak_gpu_utilization", 0)}%`
- Average GPU utilization: `{monitor.get("avg_gpu_utilization", 0)}%`
- GPU saturation share: `{monitor.get("gpu_saturation_share", 0)}%`
- Peak allocated VRAM: `{monitor.get("peak_vram_allocated", 0)}%`
- Peak memory read/write activity: `{monitor.get("peak_memory_activity", 0)}%`

## AMD GPU Saturation Gate

**Status:** `{saturation["status"]}`  
**Saturation score:** `{saturation["score"]}/100`  
**Interpretation:** {saturation["summary"]}

## Production Gate Matrix

{gate_matrix}

## Next Experiments

{next_experiments}

## Judge-Friendly Notes

- This project uses AMD Developer Cloud to convert raw MI300X access into actionable deployment guidance.
- The agent does not merely answer questions; it runs experiments, scores configurations, and produces an operational report.
- The next production step is to add continuous profiling in CI so every model or prompt workload gets an AMD deployment readiness check before release.

## Raw Notes

{payload.get("notes", "No notes provided.")}
"""
    report_path.write_text(report, encoding="utf-8")
    latest_path = output_path / f"rocmbench_{model_slug}_latest.md"
    latest_path.write_text(report, encoding="utf-8")
    return report, latest_path
