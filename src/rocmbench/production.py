from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import shlex
import socket
import sqlite3
import statistics
import subprocess
import time
from typing import Any

import pandas as pd
import requests

from .analysis import analyze_benchmark, build_deployment_gate_matrix, deployment_decision, estimate_costs, plan_next_experiments


DEFAULT_PROMPTS = [
    "Explain how ROCm helps run machine learning workloads on AMD GPUs.",
    "Write a concise deployment checklist for serving Qwen with vLLM.",
    "Summarize the business value of benchmarking inference configurations.",
    "Create a short customer-support response for a delayed enterprise AI deployment.",
]


def check_openai_endpoint(endpoint: str) -> tuple[bool, str]:
    if not endpoint or not endpoint.strip():
        return False, "Endpoint is empty. Start vLLM and enter its OpenAI-compatible base URL."
    try:
        response = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=8)
        response.raise_for_status()
    except Exception as exc:
        return False, f"Endpoint is not reachable at `{endpoint}`: {exc}"
    return True, f"Endpoint is reachable at `{endpoint}`."


def list_openai_endpoint_models(endpoint: str) -> list[str]:
    response = requests.get(f"{endpoint.rstrip('/')}/v1/models", timeout=8)
    response.raise_for_status()
    body = response.json()
    return [str(item.get("id")) for item in body.get("data", []) if item.get("id")]


def _is_local_port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, int(port))) == 0


def ensure_ssh_tunnel(
    ssh_target: str,
    ssh_key_path: str,
    local_port: int = 8000,
    remote_port: int = 8000,
) -> str:
    if _is_local_port_open(local_port):
        return f"SSH tunnel already listening on localhost:{local_port}."

    key_path = str(Path(ssh_key_path).expanduser())
    subprocess.Popen(
        [
            "ssh",
            "-i",
            key_path,
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ServerAliveInterval=30",
            "-N",
            "-L",
            f"{local_port}:localhost:{remote_port}",
            ssh_target,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(20):
        if _is_local_port_open(local_port):
            return f"Opened SSH tunnel localhost:{local_port} -> {ssh_target}:localhost:{remote_port}."
        time.sleep(0.5)
    raise RuntimeError(f"Could not open SSH tunnel to {ssh_target} on localhost:{local_port}.")


def start_remote_vllm_model(
    model: str,
    ssh_target: str,
    ssh_key_path: str,
    container_name: str,
    endpoint: str,
    max_model_len: int = 4096,
    max_num_seqs: int = 24,
    gpu_memory_utilization: float = 0.90,
    wait_seconds: int = 420,
    ensure_tunnel: bool = True,
) -> str:
    model = model.strip()
    if not model:
        raise ValueError("Model is empty.")
    if ensure_tunnel:
        tunnel_message = ensure_ssh_tunnel(ssh_target, ssh_key_path)
    else:
        tunnel_message = "SSH tunnel auto-start skipped."

    key_path = str(Path(ssh_key_path).expanduser())
    safe_model = "".join(char if char.isalnum() else "_" for char in model)
    remote_script = f"""
set -euo pipefail
cd /workspace/rocmbench
served="$(curl -fsS http://127.0.0.1:8000/v1/models 2>/dev/null || true)"
if printf "%s" "$served" | grep -Fq {shlex.quote(model)}; then
  echo "already-serving {shlex.quote(model)}"
  exit 0
fi
pkill -f "/usr/local/bin/vllm serve" 2>/dev/null || true
pkill -f "vllm.entrypoints.openai" 2>/dev/null || true
sleep 8
nohup vllm serve {shlex.quote(model)} \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --max-model-len {int(max_model_len)} \\
  --max-num-seqs {int(max_num_seqs)} \\
  --gpu-memory-utilization {float(gpu_memory_utilization):.2f} \\
  > vllm_live_{safe_model}_{int(max_model_len)}_{int(max_num_seqs)}.log 2>&1 < /dev/null &
echo $! > vllm_live.pid
echo "started {shlex.quote(model)} pid=$(cat vllm_live.pid)"
"""
    result = subprocess.run(
        [
            "ssh",
            "-i",
            key_path,
            "-o",
            "StrictHostKeyChecking=accept-new",
            ssh_target,
            "docker",
            "exec",
            "-i",
            container_name,
            "bash",
            "-s",
        ],
        input=remote_script,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Remote vLLM start failed: {result.stderr or result.stdout}")

    deadline = time.time() + max(30, int(wait_seconds))
    last_error = ""
    while time.time() < deadline:
        try:
            served_models = list_openai_endpoint_models(endpoint)
            if model in served_models:
                return f"{tunnel_message}\nRemote vLLM is serving `{model}`."
            if served_models:
                last_error = f"Endpoint currently serves: {', '.join(served_models)}"
        except Exception as exc:
            last_error = str(exc)
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting for `{model}` at {endpoint}. Last status: {last_error}")


def init_history(db_path: str | Path = "rocmbench_history.sqlite") -> None:
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                model TEXT NOT NULL,
                hardware TEXT NOT NULL,
                workload TEXT NOT NULL,
                objective TEXT NOT NULL,
                best_config TEXT NOT NULL,
                readiness_score REAL NOT NULL,
                tokens_per_second REAL NOT NULL,
                p95_latency_s REAL NOT NULL,
                error_rate REAL NOT NULL,
                cost_per_million_tokens REAL NOT NULL,
                source TEXT NOT NULL
            )
            """
        )


def record_history(
    payload: dict[str, Any],
    objective: str,
    hourly_price: float,
    source: str,
    db_path: str | Path = "rocmbench_history.sqlite",
) -> None:
    init_history(db_path)
    frame, recommendation = analyze_benchmark(payload, objective)
    best = frame.iloc[0]
    costs = estimate_costs(frame, hourly_price, 100_000, 256)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO benchmark_runs (
                captured_at, model, hardware, workload, objective, best_config,
                readiness_score, tokens_per_second, p95_latency_s, error_rate,
                cost_per_million_tokens, source
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload.get("captured_at") or datetime.now(timezone.utc).isoformat()),
                str(payload.get("model", "unknown")),
                str(payload.get("hardware", "unknown")),
                str(payload.get("workload", "unknown")),
                objective,
                recommendation.config_name,
                recommendation.score,
                float(best["tokens_per_second"]),
                float(best["p95_latency_s"]),
                float(best["error_rate"]),
                float(costs["cost_per_million_tokens"]),
                source,
            ),
        )


def load_history(db_path: str | Path = "rocmbench_history.sqlite", limit: int = 25) -> pd.DataFrame:
    init_history(db_path)
    with sqlite3.connect(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT captured_at, model, hardware, workload, objective, best_config,
                   readiness_score, tokens_per_second, p95_latency_s, error_rate,
                   cost_per_million_tokens, source
            FROM benchmark_runs
            ORDER BY id DESC
            LIMIT ?
            """,
            connection,
            params=(limit,),
        )


def assess_regression(
    current_payload: dict[str, Any],
    objective: str,
    hourly_price: float,
    db_path: str | Path = "rocmbench_history.sqlite",
) -> str:
    history = load_history(db_path, limit=100)
    frame, recommendation = analyze_benchmark(current_payload, objective)
    best = frame.iloc[0]
    costs = estimate_costs(frame, hourly_price, 100_000, 256)
    model = str(current_payload.get("model", "unknown"))
    workload = str(current_payload.get("workload", "unknown"))
    previous = history[(history["model"] == model) & (history["workload"] == workload)]

    status = "APPROVE"
    bullets = [
        f"Current best config: **{recommendation.config_name}**",
        f"Readiness score: **{recommendation.score:.1f}/100**",
        f"P95 latency: **{float(best['p95_latency_s']):.3f}s**",
        f"Error rate: **{float(best['error_rate']):.1%}**",
        f"Cost per 1M output tokens: **${costs['cost_per_million_tokens']:.2f}**",
    ]

    if previous.empty:
        return "### Deployment Decision: APPROVE\n\nNo previous matching run was found, so this run becomes the production baseline.\n\n- " + "\n- ".join(bullets)

    baseline = previous.iloc[0]
    p95_change = (float(best["p95_latency_s"]) - float(baseline["p95_latency_s"])) / max(float(baseline["p95_latency_s"]), 0.001)
    cost_change = (costs["cost_per_million_tokens"] - float(baseline["cost_per_million_tokens"])) / max(float(baseline["cost_per_million_tokens"]), 0.001)
    error_rate = float(best["error_rate"])

    if p95_change > 0.20 or error_rate > 0.01:
        status = "BLOCK"
    elif cost_change > 0.15:
        status = "WARN"

    bullets.extend(
        [
            f"Previous best config: **{baseline['best_config']}**",
            f"P95 latency change: **{p95_change:+.1%}**",
            f"Cost per 1M token change: **{cost_change:+.1%}**",
        ]
    )
    return f"### Deployment Decision: {status}\n\n- " + "\n- ".join(bullets)


def compare_payloads(payloads: dict[str, dict[str, Any]], objective: str, hourly_price: float) -> pd.DataFrame:
    rows = []
    for name, payload in payloads.items():
        frame, recommendation = analyze_benchmark(payload, objective)
        best = frame.iloc[0]
        costs = estimate_costs(frame, hourly_price, 100_000, 256)
        rows.append(
            {
                "Scenario": name,
                "Model": payload.get("model", "unknown"),
                "Workload": payload.get("workload", "unknown"),
                "Best config": recommendation.config_name,
                "Score": recommendation.score,
                "Tokens/sec": float(best["tokens_per_second"]),
                "P95 latency": float(best["p95_latency_s"]),
                "Error rate": float(best["error_rate"]),
                "Cost / 1M tokens": costs["cost_per_million_tokens"],
            }
        )
    return pd.DataFrame(rows).sort_values(["Score", "Tokens/sec"], ascending=False)


def generate_production_artifacts(
    payload: dict[str, Any],
    objective: str,
    output_dir: str | Path = "artifacts",
) -> tuple[str, Path]:
    frame, recommendation = analyze_benchmark(payload, objective)
    best = frame.iloc[0]
    decision = deployment_decision(frame, recommendation)
    gate_matrix = build_deployment_gate_matrix(frame, recommendation, {}).to_markdown(index=False)
    next_experiments = plan_next_experiments(frame, recommendation, {}).to_markdown(index=False)
    root = Path(output_dir) / "rocmbench_deployment_bundle"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    model = str(payload.get("model", "unknown-model"))
    command = recommendation.launch_hint
    (root / "vllm_start.sh").write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail
export HIP_VISIBLE_DEVICES="${{HIP_VISIBLE_DEVICES:-0}}"
{command}
""",
        encoding="utf-8",
    )
    (root / "docker-compose.yml").write_text(
        f"""services:
  vllm-amd:
    image: rocm/vllm:latest
    ipc: host
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
    ports:
      - "8000:8000"
    command: >
      vllm serve {model}
      --max-model-len {int(best["max_model_len"])}
      --max-num-seqs {int(best["concurrency"])}
      --gpu-memory-utilization 0.90
""",
        encoding="utf-8",
    )
    (root / "rocmbench-ci.yml").write_text(
        f"""name: ROCmBench AMD readiness
on: [workflow_dispatch]
jobs:
  benchmark:
    runs-on: self-hosted
    steps:
      - uses: actions/checkout@v4
      - name: Run ROCmBench
        run: |
          python scripts/run_vllm_benchmark.py \\
            --endpoint http://localhost:8000 \\
            --model {model} \\
            --concurrency 1 4 8 16 \\
            --requests 32 \\
            --output benchmark_results/ci_amd_run.json
""",
        encoding="utf-8",
    )
    (root / "DEPLOYMENT_DECISION.md").write_text(
        f"""# Deployment Decision

Decision: {decision["status"]}

Recommended config: `{recommendation.config_name}`

```bash
{command}
```

Reason: {recommendation.rationale}

## Production Gate Matrix

{gate_matrix}

## Next Experiments

{next_experiments}
""",
        encoding="utf-8",
    )
    archive_base = Path(output_dir) / "rocmbench_deployment_bundle"
    archive = shutil.make_archive(str(archive_base), "zip", root)
    summary = f"Generated production bundle with `vllm_start.sh`, `docker-compose.yml`, CI workflow, and deployment decision for `{recommendation.config_name}`."
    return summary, Path(archive)


def _call_endpoint(endpoint: str, model: str, prompt: str, max_tokens: int, temperature: float) -> dict[str, float]:
    started = time.perf_counter()
    response = requests.post(
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
        "completion_tokens": float(usage.get("completion_tokens") or max_tokens),
    }


def run_live_benchmark(
    endpoint: str,
    model: str,
    workload: str,
    concurrency_values: list[int],
    requests_per_config: int,
    max_tokens: int = 256,
    temperature: float = 0.2,
) -> dict[str, Any]:
    runs = []
    for concurrency in concurrency_values:
        measurements = []
        errors = 0
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
            futures = [
                pool.submit(_call_endpoint, endpoint, model, DEFAULT_PROMPTS[index % len(DEFAULT_PROMPTS)], max_tokens, temperature)
                for index in range(requests_per_config)
            ]
            for future in as_completed(futures):
                try:
                    measurements.append(future.result())
                except Exception:
                    errors += 1
        total_elapsed = time.perf_counter() - started
        latencies = [float(item["latency_s"]) for item in measurements] or [0.0]
        completion_tokens = sum(float(item["completion_tokens"]) for item in measurements)
        p95 = sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
        runs.append(
            {
                "config_name": f"live-concurrency-{concurrency}",
                "concurrency": concurrency,
                "max_model_len": 4096,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "quantization": "none",
                "avg_latency_s": round(statistics.mean(latencies), 3),
                "p95_latency_s": round(p95, 3),
                "tokens_per_second": round(completion_tokens / max(total_elapsed, 0.001), 2),
                "requests_per_minute": round(len(measurements) / max(total_elapsed, 0.001) * 60, 2),
                "gpu_memory_gb": 0.0,
                "error_rate": round(errors / max(requests_per_config, 1), 4),
            }
        )

    return {
        "project": "ROCmBench Agent",
        "hardware": "AMD Instinct GPU endpoint",
        "backend": "ROCm + vLLM",
        "model": model,
        "workload": workload,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "notes": f"Live benchmark executed from UI against {endpoint}.",
        "runs": runs,
    }


def save_payload(payload: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def search_huggingface_models(query: str, task: str = "text-generation", limit: int = 20) -> list[str]:
    search = (query or "").strip()
    params: dict[str, Any] = {
        "limit": max(1, min(int(limit), 50)),
        "sort": "downloads",
        "direction": -1,
        "full": "false",
    }
    if search:
        params["search"] = search
    if task and task != "any":
        params["pipeline_tag"] = task

    try:
        response = requests.get("https://huggingface.co/api/models", params=params, timeout=12)
        response.raise_for_status()
        models = response.json()
    except Exception:
        fallback = [
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "meta-llama/Llama-3.1-8B-Instruct",
            "mistralai/Mistral-7B-Instruct-v0.3",
        ]
        return [item for item in fallback if search.lower() in item.lower()] or fallback

    results = []
    for model in models:
        model_id = model.get("modelId") or model.get("id")
        if model_id:
            results.append(str(model_id))
    return results or ["Qwen/Qwen2.5-7B-Instruct"]
