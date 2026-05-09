from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PROMPTS = [
    "Explain how ROCm helps run machine learning workloads on AMD GPUs.",
    "Write a concise deployment checklist for serving Qwen with vLLM.",
    "Summarize the business value of benchmarking inference configurations.",
    "Create a short customer-support response for a delayed enterprise AI deployment.",
]


async def call_openai_compatible_endpoint(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = await asyncio.to_thread(
        requests.post,
        f"{endpoint.rstrip('/')}/v1/chat/completions",
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage", {})
    return {
        "latency_s": elapsed,
        "completion_tokens": usage.get("completion_tokens", max_tokens),
        "total_tokens": usage.get("total_tokens", max_tokens),
    }


async def run_config(args: argparse.Namespace, concurrency: int) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(concurrency)
    errors = 0
    measurements: list[dict[str, Any]] = []

    async def worker(index: int) -> None:
        nonlocal errors
        prompt = PROMPTS[index % len(PROMPTS)]
        async with semaphore:
            try:
                result = await call_openai_compatible_endpoint(
                    args.endpoint,
                    args.model,
                    prompt,
                    args.max_tokens,
                    args.temperature,
                )
                measurements.append(result)
            except Exception:
                errors += 1

    started = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(args.requests)))
    total_elapsed = time.perf_counter() - started

    latencies = [m["latency_s"] for m in measurements] or [0.0]
    completion_tokens = sum(int(m["completion_tokens"]) for m in measurements)
    p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]

    return {
        "config_name": f"concurrency-{concurrency}",
        "concurrency": concurrency,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "quantization": args.quantization,
        "avg_latency_s": round(statistics.mean(latencies), 3),
        "p95_latency_s": round(p95, 3),
        "tokens_per_second": round(completion_tokens / max(total_elapsed, 0.001), 2),
        "requests_per_minute": round(len(measurements) / max(total_elapsed, 0.001) * 60, 2),
        "gpu_memory_gb": args.gpu_memory_gb,
        "error_rate": round(errors / max(args.requests, 1), 4),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark an OpenAI-compatible vLLM server on AMD GPUs.")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="benchmark_results/amd_run.json")
    parser.add_argument("--hardware", default="AMD Instinct MI300X")
    parser.add_argument("--workload", default="Prompt workload")
    parser.add_argument("--requests", type=int, default=24)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--quantization", default="none")
    parser.add_argument("--gpu-memory-gb", type=float, default=0.0, help="Optional observed VRAM usage for this run.")
    args = parser.parse_args()

    runs = []
    for concurrency in args.concurrency:
        runs.append(await run_config(args, concurrency))

    payload = {
        "project": "ROCmBench Agent",
        "hardware": args.hardware,
        "backend": "ROCm + vLLM",
        "model": args.model,
        "workload": args.workload,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "notes": "Measured through the OpenAI-compatible vLLM API. Fill gpu_memory_gb from rocm-smi for stronger final reporting.",
        "runs": runs,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    asyncio.run(main())
