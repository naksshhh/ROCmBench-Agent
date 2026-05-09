from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from src.rocmbench import (
    analyze_benchmark,
    assess_regression,
    assess_gpu_saturation,
    build_deployment_gate_matrix,
    build_report,
    check_openai_endpoint,
    compare_payloads,
    deployment_decision,
    estimate_costs,
    generate_production_artifacts,
    list_openai_endpoint_models,
    load_history,
    parse_rocm_monitor,
    parse_rocm_monitor_series,
    plan_next_experiments,
    record_history,
    run_live_benchmark,
    save_payload,
    search_huggingface_models,
    start_remote_vllm_model,
    summarize_impact,
)


ROOT = Path(__file__).parent
BENCHMARKS = {
    "Real AMD MI300X run": ROOT / "benchmark_results" / "amd_qwen25_7b.json",
    "Real Qwen 7B concurrency ceiling": ROOT / "benchmark_results" / "qwen25_7b_concurrency32.json",
    "Real Qwen 7B 8192 context boundary": ROOT / "benchmark_results" / "qwen25_7b_long_context_8192.json",
    "Real Qwen 7B soak stability": ROOT / "benchmark_results" / "qwen25_7b_soak_concurrency16.json",
    "Real Qwen 3B comparison": ROOT / "benchmark_results" / "qwen25_3b_comparison.json",
    "Sample MI300X demo": ROOT / "benchmark_results" / "sample_qwen25_7b_mi300x.json",
    "Long-context tradeoff sample": ROOT / "benchmark_results" / "sample_long_context_tradeoff.json",
}
MONITOR_LOGS = {
    "Real AMD MI300X run": ROOT / "benchmark_results" / "amd_qwen25_7b_rocm_monitor.log",
    "Real Qwen 7B concurrency ceiling": ROOT / "benchmark_results" / "qwen25_7b_concurrency32_rocm_monitor.log",
    "Real Qwen 7B 8192 context boundary": ROOT / "benchmark_results" / "qwen25_7b_long_context_8192_rocm_monitor.log",
    "Real Qwen 7B soak stability": ROOT / "benchmark_results" / "qwen25_7b_soak_concurrency16_rocm_monitor.log",
    "Real Qwen 3B comparison": ROOT / "benchmark_results" / "qwen25_3b_comparison_rocm_monitor.log",
}
HISTORY_DB = ROOT / "rocmbench_history.sqlite"
DEFAULT_SSH_TARGET = os.getenv("ROCMBENCH_SSH_TARGET", "root@YOUR_DROPLET_IP")
DEFAULT_SSH_KEY = os.getenv("ROCMBENCH_SSH_KEY", "~/.ssh/your_key")
DEFAULT_CONTAINER = os.getenv("ROCMBENCH_CONTAINER", "rocm")
HF_TASKS = [
    "text-generation",
    "text2text-generation",
    "question-answering",
    "summarization",
    "automatic-speech-recognition",
    "image-text-to-text",
    "visual-question-answering",
    "any",
]
WORKLOAD_TYPES = [
    "Customer-support chat",
    "Code assistant",
    "RAG question answering",
    "Long-context analysis",
    "Summarization",
    "Multimodal assistant",
    "Production smoke test",
]

AMD_CSS = """
:root {
  --amd-black: #050506;
  --amd-panel: #11151d;
  --amd-panel-2: #171d28;
  --amd-border: rgba(255, 255, 255, 0.11);
  --amd-red: #ed1c24;
  --amd-red-2: #ff4b3e;
  --amd-cyan: #00c7d4;
  --amd-text: #f7f7f8;
  --amd-muted: #aab1c0;
}

body,
.gradio-container {
  background:
    radial-gradient(circle at 78% 8%, rgba(237, 28, 36, 0.28), transparent 28rem),
    radial-gradient(circle at 16% 0%, rgba(0, 199, 212, 0.13), transparent 22rem),
    linear-gradient(180deg, #030405 0%, #070a10 42%, #090d16 100%) !important;
  color: var(--amd-text) !important;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

.gradio-container {
  max-width: 1500px !important;
}

footer {
  display: none !important;
}

.amd-hero {
  position: relative;
  overflow: hidden;
  min-height: 310px;
  border: 1px solid rgba(237, 28, 36, 0.36);
  background:
    linear-gradient(90deg, rgba(0, 0, 0, 0.94) 0%, rgba(9, 10, 14, 0.76) 54%, rgba(237, 28, 36, 0.20) 100%),
    repeating-linear-gradient(90deg, transparent 0 42px, rgba(255, 255, 255, 0.035) 42px 43px),
    repeating-linear-gradient(0deg, transparent 0 42px, rgba(255, 255, 255, 0.025) 42px 43px);
  border-radius: 0;
  padding: 34px 42px;
  box-shadow: 0 24px 90px rgba(0, 0, 0, 0.55), inset 0 0 80px rgba(237, 28, 36, 0.06);
}

.amd-hero:before {
  content: "";
  position: absolute;
  inset: 0;
  background:
    linear-gradient(135deg, transparent 0 54%, rgba(237, 28, 36, 0.72) 54.2%, transparent 55%),
    linear-gradient(155deg, transparent 0 62%, rgba(255, 75, 62, 0.50) 62.2%, transparent 63%),
    radial-gradient(circle at 78% 42%, rgba(237, 28, 36, 0.52), transparent 11rem);
  opacity: 0.86;
  pointer-events: none;
}

.amd-hero-inner {
  position: relative;
  z-index: 1;
  max-width: 980px;
}

.amd-logo-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 12px;
  margin-bottom: 42px;
  color: rgba(255, 255, 255, 0.78);
  font-size: 13px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.amd-logo {
  color: #ffffff;
  font-weight: 900;
  font-size: 20px;
  letter-spacing: 0;
}

.amd-partner {
  border: 1px solid rgba(255, 255, 255, 0.14);
  color: rgba(255, 255, 255, 0.74);
  padding: 5px 9px;
  background: rgba(255, 255, 255, 0.035);
}

.amd-eyebrow {
  color: var(--amd-red-2);
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 13px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  margin-bottom: 14px;
}

.amd-hero h1 {
  color: #ffffff;
  font-size: clamp(48px, 6vw, 88px);
  line-height: 0.92;
  letter-spacing: 0;
  font-weight: 800;
  margin: 0 0 18px;
  text-transform: uppercase;
}

.amd-hero p {
  max-width: 760px;
  color: rgba(255, 255, 255, 0.78);
  font-size: 21px;
  line-height: 1.45;
  margin: 0;
}

.amd-kpis {
  position: relative;
  z-index: 1;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 1px;
  max-width: 990px;
  margin-top: 34px;
  background: rgba(255, 255, 255, 0.18);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.amd-kpi {
  background: rgba(5, 5, 6, 0.82);
  padding: 16px 18px;
}

.amd-kpi b {
  display: block;
  color: #ffffff;
  font-size: 24px;
  line-height: 1;
  margin-bottom: 8px;
}

.amd-kpi span {
  color: var(--amd-muted);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.amd-control-panel,
.amd-output-panel,
.amd-chart-panel,
.amd-table-panel {
  border: 1px solid var(--amd-border) !important;
  background: linear-gradient(180deg, rgba(23, 29, 40, 0.92), rgba(12, 15, 22, 0.96)) !important;
  box-shadow: 0 18px 48px rgba(0, 0, 0, 0.28) !important;
  border-radius: 6px !important;
}

.amd-output-panel {
  padding: 18px !important;
}

.amd-section-card {
  border: 1px solid rgba(255, 255, 255, 0.11);
  background: rgba(255, 255, 255, 0.035);
  border-radius: 6px;
  padding: 14px 16px;
}

.amd-metric-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
  gap: 12px;
  margin: 4px 0 16px;
}

.amd-metric {
  min-height: 116px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background:
    linear-gradient(135deg, rgba(237, 28, 36, 0.18), transparent 48%),
    rgba(255, 255, 255, 0.04);
  border-radius: 6px;
  padding: 16px;
}

.amd-metric .label {
  color: var(--amd-muted);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 14px;
}

.amd-metric .value {
  color: #ffffff;
  font-size: 30px;
  font-weight: 800;
  line-height: 1;
}

.amd-metric .note {
  color: rgba(255, 255, 255, 0.64);
  margin-top: 10px;
  font-size: 13px;
}

.amd-section-card h3,
.amd-output-panel h3 {
  color: #ffffff !important;
  margin-top: 0 !important;
  letter-spacing: 0 !important;
}

.amd-section-card strong,
.amd-output-panel strong {
  color: #ffffff !important;
}

.amd-section-card li::marker,
.amd-output-panel li::marker {
  color: var(--amd-red-2);
}

.amd-section-card code,
.amd-output-panel code {
  background: rgba(0, 0, 0, 0.42) !important;
  border: 1px solid rgba(237, 28, 36, 0.24) !important;
  color: #ffd8d8 !important;
}

.amd-run-button button,
button.primary {
  background: linear-gradient(90deg, var(--amd-red), #ff673d) !important;
  border: 1px solid rgba(255, 255, 255, 0.18) !important;
  color: white !important;
  font-weight: 800 !important;
  min-height: 48px !important;
  border-radius: 4px !important;
  box-shadow: 0 16px 40px rgba(237, 28, 36, 0.26) !important;
}

label,
.label-wrap span {
  color: #ffffff !important;
  font-weight: 750 !important;
}

input,
textarea,
select,
.wrap,
.container,
.input-container,
.block {
  border-radius: 6px !important;
}

textarea,
input,
.wrap.default,
.secondary-wrap {
  background: rgba(255, 255, 255, 0.055) !important;
  color: var(--amd-text) !important;
  border-color: rgba(255, 255, 255, 0.10) !important;
}

.amd-chart-panel {
  padding: 10px !important;
}

.amd-chart-panel .js-plotly-plot,
.amd-chart-panel .plot-container,
.amd-chart-panel svg.main-svg {
  background: transparent !important;
}

.amd-table-panel table {
  font-size: 13px !important;
}

.amd-table-panel th {
  background: #11151d !important;
  color: #ffffff !important;
}

.amd-table-panel td {
  background: #090d16 !important;
  color: #e7e9ef !important;
}

@media (max-width: 900px) {
  .amd-hero {
    padding: 28px 22px;
  }

  .amd-kpis,
  .amd-metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .amd-hero h1 {
    font-size: 44px;
  }
}
"""


def load_payload(scenario: str, uploaded_file: str | None) -> dict[str, Any]:
    source = Path(uploaded_file) if uploaded_file else BENCHMARKS[scenario]
    with source.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _monitor_path(scenario: str, uploaded_log: str | None) -> Path | None:
    if uploaded_log:
        return Path(uploaded_log)
    return MONITOR_LOGS.get(scenario)


def _format_monitor(summary: dict[str, Any]) -> str:
    return f"""
### Live ROCm Monitor

- Status: **{summary["status"]}**
- ROCm samples: **{summary["samples"]}**
- Peak GPU utilization: **{summary["peak_gpu_utilization"]}%**
- Average GPU utilization: **{summary.get("avg_gpu_utilization", 0)}%**
- GPU saturation share: **{summary.get("gpu_saturation_share", 0)}%**
- Peak allocated VRAM: **{summary["peak_vram_allocated"]}%**
- Peak memory read/write activity: **{summary["peak_memory_activity"]}%**
"""


def _current_payload(scenario: str, uploaded_file: str | None) -> dict[str, Any]:
    return load_payload(scenario, uploaded_file)


def _parse_concurrency(raw: str) -> list[int]:
    values = []
    for part in raw.replace(",", " ").split():
        try:
            value = int(part)
        except ValueError:
            continue
        if value > 0:
            values.append(value)
    return values or [1, 4, 8, 16]


def search_models_for_ui(query: str, task: str) -> tuple[Any, str, str]:
    models = search_huggingface_models(query, task, limit=20)
    selected = models[0]
    status = f"Found {len(models)} Hugging Face model suggestion(s) for task `{task}`."
    return gr.update(choices=models, value=selected), selected, status


def save_current_to_history(
    scenario: str,
    uploaded_file: str | None,
    objective: str,
    hourly_price: float,
) -> tuple[pd.DataFrame, str]:
    payload = _current_payload(scenario, uploaded_file)
    record_history(payload, objective, hourly_price, f"scenario:{scenario}", HISTORY_DB)
    return load_history(HISTORY_DB), assess_regression(payload, objective, hourly_price, HISTORY_DB)


def refresh_history() -> pd.DataFrame:
    return load_history(HISTORY_DB)


def check_current_regression(
    scenario: str,
    uploaded_file: str | None,
    objective: str,
    hourly_price: float,
) -> str:
    payload = _current_payload(scenario, uploaded_file)
    return assess_regression(payload, objective, hourly_price, HISTORY_DB)


def compare_models(objective: str, hourly_price: float) -> pd.DataFrame:
    payloads = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in BENCHMARKS.items()}
    return compare_payloads(payloads, objective, hourly_price)


def generate_artifacts(
    scenario: str,
    uploaded_file: str | None,
    objective: str,
) -> tuple[str, str]:
    payload = _current_payload(scenario, uploaded_file)
    summary, archive = generate_production_artifacts(payload, objective, ROOT / "artifacts")
    return f"### Production Config Generator\n\n{summary}", str(archive)


def refresh_live_monitor(scenario: str, uploaded_log: str | None) -> str:
    return _format_monitor(parse_rocm_monitor(_monitor_path(scenario, uploaded_log)))


def refresh_monitor_panel(scenario: str, uploaded_log: str | None) -> tuple[str, Any]:
    return refresh_live_monitor(scenario, uploaded_log), rocm_monitor_chart(scenario, uploaded_log)


def rocm_monitor_chart(scenario: str, uploaded_log: str | None) -> Any:
    series = parse_rocm_monitor_series(_monitor_path(scenario, uploaded_log))
    chart = go.Figure()
    if series.empty:
        chart.add_annotation(
            text="Upload or select a ROCm monitor log to show GPU evidence over time.",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="#f4f6fb", size=16),
        )
    else:
        chart.add_trace(
            go.Scatter(
                x=series["sample"],
                y=series["gpu_utilization"],
                name="GPU utilization",
                mode="lines",
                line=dict(color="#ed1c24", width=3),
            )
        )
        chart.add_trace(
            go.Scatter(
                x=series["sample"],
                y=series["vram_allocated"],
                name="VRAM allocated",
                mode="lines",
                line=dict(color="#ffb000", width=2),
            )
        )
        chart.add_trace(
            go.Scatter(
                x=series["sample"],
                y=series["memory_activity"],
                name="Memory activity",
                mode="lines",
                line=dict(color="#00c7d4", width=2),
            )
        )
    chart.update_layout(
        template="plotly_dark",
        height=360,
        margin=dict(l=24, r=24, t=52, b=40),
        paper_bgcolor="#090d16",
        plot_bgcolor="#0d1320",
        font=dict(color="#f4f6fb"),
        title=dict(text="ROCm SMI evidence over time", font=dict(size=18, color="#ffffff")),
        xaxis=dict(title="Sample", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Percent", range=[0, 105], gridcolor="rgba(255,255,255,0.08)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return chart


def run_live_endpoint_benchmark(
    endpoint: str,
    selected_model: str,
    manual_model: str,
    workload_type: str,
    workload_detail: str,
    concurrency_csv: str,
    request_count: int,
    auto_load_model: bool,
    ssh_target: str,
    ssh_key_path: str,
    container_name: str,
    serve_max_model_len: int,
    serve_max_num_seqs: int,
    objective: str,
    hourly_price: float,
) -> tuple[str, str, pd.DataFrame, str, pd.DataFrame]:
    concurrency_values = _parse_concurrency(concurrency_csv)
    model = (manual_model or selected_model or "Qwen/Qwen2.5-7B-Instruct").strip()
    workload = f"{workload_type}: {workload_detail}".strip(": ")
    load_message = ""
    if auto_load_model:
        try:
            load_message = start_remote_vllm_model(
                model=model,
                ssh_target=ssh_target or DEFAULT_SSH_TARGET,
                ssh_key_path=ssh_key_path or DEFAULT_SSH_KEY,
                container_name=container_name or DEFAULT_CONTAINER,
                endpoint=endpoint,
                max_model_len=int(serve_max_model_len or 4096),
                max_num_seqs=max(int(serve_max_num_seqs or 1), max(concurrency_values)),
                ensure_tunnel=endpoint.startswith("http://localhost") or endpoint.startswith("http://127.0.0.1"),
            )
        except Exception as exc:
            failed_frame = pd.DataFrame(
                columns=[
                    "config_name",
                    "concurrency",
                    "tokens_per_second",
                    "avg_latency_s",
                    "p95_latency_s",
                    "error_rate",
                ]
            )
            status = f"""
### Live Benchmark Blocked

Could not load **{model}** on the AMD droplet.

`{exc}`

Check the SSH target/key/container settings, or disable auto-load and serve the model manually.
"""
            return status, None, failed_frame, "### Deployment Decision: BLOCK\n\nLive benchmark did not run because model loading failed.", load_history(HISTORY_DB)

    endpoint_ok, endpoint_message = check_openai_endpoint(endpoint)
    if not endpoint_ok:
        failed_frame = pd.DataFrame(
            columns=[
                "config_name",
                "concurrency",
                "tokens_per_second",
                "avg_latency_s",
                "p95_latency_s",
                "error_rate",
            ]
        )
        status = f"""
### Live Benchmark Blocked

{endpoint_message}

Start or tunnel the vLLM server, then rerun the live benchmark.

For the remote AMD droplet, `http://localhost:8000` only works if the app is running on the droplet or you have an SSH tunnel from your Mac to the droplet.
"""
        return status, None, failed_frame, "### Deployment Decision: BLOCK\n\nLive benchmark did not run because the endpoint was unreachable.", load_history(HISTORY_DB)

    served_models = list_openai_endpoint_models(endpoint)
    if served_models and model not in served_models:
        failed_frame = pd.DataFrame(
            columns=[
                "config_name",
                "concurrency",
                "tokens_per_second",
                "avg_latency_s",
                "p95_latency_s",
                "error_rate",
            ]
        )
        served = ", ".join(f"`{item}`" for item in served_models)
        status = f"""
### Live Benchmark Blocked

The endpoint is reachable, but it is not serving **{model}**.

Currently served model(s): {served}

Start vLLM with the selected model, or set the selected/manual model field to the model returned by `/v1/models`.
"""
        return status, None, failed_frame, "### Deployment Decision: BLOCK\n\nLive benchmark did not run because the selected model does not match the served vLLM model.", load_history(HISTORY_DB)

    payload = run_live_benchmark(
        endpoint=endpoint,
        model=model,
        workload=workload,
        concurrency_values=concurrency_values,
        requests_per_config=max(1, int(request_count)),
    )
    stamp = payload["captured_at"].replace(":", "").replace("-", "").split(".")[0]
    output = save_payload(payload, ROOT / "benchmark_results" / f"live_{stamp}.json")
    record_history(payload, objective, hourly_price, "live-ui", HISTORY_DB)
    frame, recommendation = analyze_benchmark(payload, objective)
    decision = deployment_decision(frame, recommendation)
    status = f"""
### Live Benchmark Complete

- Endpoint: **{endpoint}**
- Model: **{model}**
- Model load: **{load_message or "already available"}**
- Tested concurrency: **{", ".join(str(item) for item in concurrency_values)}**
- Recommended config: **{recommendation.config_name}**
- Decision: **{decision["status"]}**
- Output JSON: `{output}`
"""
    return status, str(output), frame, assess_regression(payload, objective, hourly_price, HISTORY_DB), load_history(HISTORY_DB)


def load_selected_model_on_droplet(
    endpoint: str,
    selected_model: str,
    manual_model: str,
    ssh_target: str,
    ssh_key_path: str,
    container_name: str,
    serve_max_model_len: int,
    serve_max_num_seqs: int,
) -> str:
    model = (manual_model or selected_model or "Qwen/Qwen2.5-7B-Instruct").strip()
    try:
        message = start_remote_vllm_model(
            model=model,
            ssh_target=ssh_target or DEFAULT_SSH_TARGET,
            ssh_key_path=ssh_key_path or DEFAULT_SSH_KEY,
            container_name=container_name or DEFAULT_CONTAINER,
            endpoint=endpoint,
            max_model_len=int(serve_max_model_len or 4096),
            max_num_seqs=int(serve_max_num_seqs or 24),
            ensure_tunnel=endpoint.startswith("http://localhost") or endpoint.startswith("http://127.0.0.1"),
        )
    except Exception as exc:
        return f"### Model Load Failed\n\nCould not load **{model}**.\n\n`{exc}`"
    return f"### Model Ready\n\n{message}"


def dashboard(
    scenario: str,
    uploaded_file: str | None,
    uploaded_log: str | None,
    objective: str,
    hourly_price: float,
    requests_per_day: int,
    output_tokens_per_request: int,
) -> tuple[str, pd.DataFrame, Any, str, str, str, str, str, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    payload = load_payload(scenario, uploaded_file)
    frame, recommendation = analyze_benchmark(payload, objective)
    impact = summarize_impact(frame)
    costs = estimate_costs(frame, hourly_price, requests_per_day, output_tokens_per_request)
    monitor_summary = parse_rocm_monitor(_monitor_path(scenario, uploaded_log))
    decision = deployment_decision(frame, recommendation)
    saturation = assess_gpu_saturation(frame, monitor_summary)
    next_experiments = plan_next_experiments(frame, recommendation, monitor_summary)
    gate_matrix = build_deployment_gate_matrix(frame, recommendation, monitor_summary)

    chart_frame = frame.sort_values("concurrency", ascending=True)
    configs = chart_frame["config_name"].astype(str).tolist()
    throughput = chart_frame["tokens_per_second"].astype(float).tolist()
    p95_latency = chart_frame["p95_latency_s"].astype(float).tolist()
    readiness = chart_frame["amd_readiness_score"].astype(float).tolist()
    bar_colors = ["#5b1618" if score < 40 else "#b71922" if score < 80 else "#ff3b30" for score in readiness]
    chart = go.Figure()
    chart.add_trace(
        go.Bar(
            x=configs,
            y=throughput,
            name="Tokens/sec",
            marker=dict(color=bar_colors, line=dict(color="rgba(255,255,255,0.45)", width=1)),
            customdata=list(zip(readiness, p95_latency, chart_frame["error_rate"].astype(float).tolist())),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Throughput: %{y:,.2f} tok/s<br>"
                "AMD readiness: %{customdata[0]:.1f}<br>"
                "P95 latency: %{customdata[1]:.3f}s<br>"
                "Error rate: %{customdata[2]:.1%}<extra></extra>"
            ),
        )
    )
    chart.add_trace(
        go.Scatter(
            x=configs,
            y=p95_latency,
            name="P95 latency",
            yaxis="y2",
            mode="lines+markers",
            line=dict(color="#ffb000", width=3),
            marker=dict(size=9, color="#ffb000", line=dict(color="#111111", width=1)),
            hovertemplate="<b>%{x}</b><br>P95 latency: %{y:.3f}s<extra></extra>",
        )
    )
    chart.update_layout(
        template="plotly_dark",
        height=430,
        margin=dict(l=24, r=24, t=58, b=40),
        paper_bgcolor="#090d16",
        plot_bgcolor="#0d1320",
        font=dict(color="#f4f6fb"),
        title=dict(text="Throughput and latency by serving config", font=dict(size=18, color="#ffffff")),
        xaxis=dict(title="Serving config", gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(237,28,36,0.35)"),
        yaxis=dict(
            title="Throughput (tokens/sec)",
            gridcolor="rgba(255,255,255,0.09)",
            zerolinecolor="rgba(237,28,36,0.35)",
        ),
        yaxis2=dict(
            title="P95 latency (seconds)",
            overlaying="y",
            side="right",
            gridcolor="rgba(255,255,255,0)",
            zeroline=False,
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        bargap=0.32,
    )

    recommendation_md = f"""
### Recommendation

**{recommendation.config_name}** wins for **{objective}** with an AMD readiness score of **{recommendation.score:.1f}/100**.

**Deployment decision:** **{decision["status"]}**  
{decision["reason"]}

{recommendation.rationale}

```bash
{recommendation.launch_hint}
```
"""
    metric_cards = f"""
<div class="amd-metric-grid">
  <div class="amd-metric">
    <div class="label">AMD Readiness</div>
    <div class="value">{recommendation.score:.1f}</div>
    <div class="note">Recommended: {recommendation.config_name}</div>
  </div>
  <div class="amd-metric">
    <div class="label">Throughput Lift</div>
    <div class="value">{impact["throughput_gain"]:.1f}x</div>
    <div class="note">{impact["baseline_tokens_per_second"]:.0f} -> {impact["recommended_tokens_per_second"]:.0f} tok/s</div>
  </div>
  <div class="amd-metric">
    <div class="label">GPU Saturation</div>
    <div class="value">{saturation["score"]:.1f}</div>
    <div class="note">{saturation["status"]}: peak {monitor_summary["peak_gpu_utilization"]}%</div>
  </div>
  <div class="amd-metric">
    <div class="label">ROCm Evidence</div>
    <div class="value">{monitor_summary["peak_gpu_utilization"]}%</div>
    <div class="note">{monitor_summary["samples"]} samples, {monitor_summary.get("gpu_saturation_share", 0)}% saturated</div>
  </div>
  <div class="amd-metric">
    <div class="label">Cost / 1M Tokens</div>
    <div class="value">${costs["cost_per_million_tokens"]:.2f}</div>
    <div class="note">At ${hourly_price:.2f}/GPU-hour</div>
  </div>
</div>
"""
    report_text, report_path = build_report(
        payload,
        objective,
        monitor_summary=monitor_summary,
        hourly_price=hourly_price,
        requests_per_day=requests_per_day,
        output_tokens_per_request=output_tokens_per_request,
    )
    metadata = (
        f"Model: {payload.get('model', 'unknown')}\n"
        f"Hardware: {payload.get('hardware', 'unknown')}\n"
        f"Backend: {payload.get('backend', 'unknown')}\n"
        f"Workload: {payload.get('workload', 'unknown')}"
    )
    proof_md = f"""
### ROCm Proof

- Status: **{monitor_summary["status"]}**
- ROCm samples: **{monitor_summary["samples"]}**
- Peak GPU utilization: **{monitor_summary["peak_gpu_utilization"]}%**
- Average GPU utilization: **{monitor_summary.get("avg_gpu_utilization", 0)}%**
- GPU saturation share: **{monitor_summary.get("gpu_saturation_share", 0)}%**
- Peak allocated VRAM: **{monitor_summary["peak_vram_allocated"]}%**
- Peak memory read/write activity: **{monitor_summary["peak_memory_activity"]}%**
"""
    saturation_md = f"""
### AMD GPU Saturation Gate

**{saturation["status"]}** with a saturation score of **{saturation["score"]:.1f}/100**.

{saturation["summary"]}

- Peak GPU utilization: **{saturation["peak_gpu_utilization"]:.0f}%**
- Average GPU utilization: **{saturation["avg_gpu_utilization"]:.1f}%**
- Saturated samples: **{saturation["gpu_saturation_share"]:.1f}%**
- Peak VRAM allocated: **{saturation["peak_vram_allocated"]:.0f}%**
"""
    impact_md = f"""
### Before / After

Baseline **{impact["baseline_config"]}** -> recommended **{impact["recommended_config"]}**

- Throughput: **{impact["baseline_tokens_per_second"]:.2f} -> {impact["recommended_tokens_per_second"]:.2f} tok/s**
- Throughput lift: **{impact["throughput_gain"]:.1f}x**
- P95 latency: **{impact["baseline_p95_latency_s"]:.3f}s -> {impact["recommended_p95_latency_s"]:.3f}s**
- Recommended error rate: **{impact["recommended_error_rate"]:.1%}**
"""
    score_breakdown = pd.DataFrame(
        [
            {"Dimension": key.replace("_", " ").title(), "Score": value}
            for key, value in recommendation.score_breakdown.items()
        ]
    )
    cost_md = f"""
### Cost Estimate

At **${hourly_price:.2f}/GPU-hour**, the recommended config can serve about **{costs["tokens_per_hour"]:,.0f} output tokens/hour**.

- Expected daily tokens: **{costs["expected_tokens_per_day"]:,.0f}**
- Estimated GPU time/day: **{costs["hours_per_day"]:.2f} hours**
- Estimated serving cost/day: **${costs["daily_cost"]:.2f}**
- Estimated cost per 1M output tokens: **${costs["cost_per_million_tokens"]:.2f}**
"""
    visible_columns = [
        "config_name",
        "amd_readiness_score",
        "concurrency",
        "tokens_per_second",
        "avg_latency_s",
        "p95_latency_s",
        "error_rate",
    ]
    return (
        metric_cards,
        frame[visible_columns],
        chart,
        recommendation_md,
        metadata,
        proof_md,
        impact_md,
        saturation_md,
        score_breakdown,
        gate_matrix,
        next_experiments,
        cost_md,
        str(report_path),
    )


with gr.Blocks(
    title="ROCmBench Agent",
    theme=gr.themes.Base(primary_hue="red", neutral_hue="slate"),
    css=AMD_CSS,
) as demo:
    gr.HTML(
        """
        <section class="amd-hero">
          <div class="amd-hero-inner">
            <div class="amd-logo-row">
              <span class="amd-logo">AMD</span>
              <span>x</span>
              <span class="amd-partner">LABLAB</span>
              <span class="amd-partner">QWEN</span>
              <span class="amd-partner">HUGGING FACE</span>
              <span class="amd-partner">ROCm</span>
              <span class="amd-partner">vLLM</span>
            </div>
            <div class="amd-eyebrow">AMD Developer Hackathon</div>
            <h1>ROCmBench Agent</h1>
            <p>Benchmark, rank, and explain open-source LLM deployment settings on AMD Instinct GPUs. Built for high-performance AI apps on AMD Developer Cloud.</p>
            <div class="amd-kpis">
              <div class="amd-kpi"><b>2,775.85</b><span>tokens / second</span></div>
              <div class="amd-kpi"><b>1.471s</b><span>p95 latency</span></div>
              <div class="amd-kpi"><b>100%</b><span>peak GPU utilization</span></div>
              <div class="amd-kpi"><b>13.1x</b><span>throughput lift</span></div>
            </div>
          </div>
        </section>
        """
    )
    with gr.Row():
        with gr.Column(scale=1, elem_classes=["amd-control-panel"]):
            scenario = gr.Dropdown(
                list(BENCHMARKS.keys()),
                value="Real AMD MI300X run",
                label="Demo scenario",
            )
            benchmark_file = gr.File(
                label="Benchmark JSON override",
                file_types=[".json"],
                type="filepath",
            )
            monitor_file = gr.File(
                label="ROCm monitor log override",
                file_types=[".log", ".txt"],
                type="filepath",
            )
            objective = gr.Radio(
                ["balanced", "lowest latency", "max throughput", "lowest memory"],
                value="balanced",
                label="Optimization goal",
            )
            hourly_price = gr.Number(value=4.0, label="GPU hourly price ($)")
            requests_per_day = gr.Number(value=100000, precision=0, label="Requests per day")
            output_tokens_per_request = gr.Number(value=256, precision=0, label="Output tokens / request")
            run_button = gr.Button("Analyze AMD Run", variant="primary", elem_classes=["amd-run-button"])
            metadata = gr.Textbox(label="Run metadata", lines=5)
        with gr.Column(scale=2, elem_classes=["amd-output-panel"]):
            summary_cards = gr.HTML()
            recommendation = gr.Markdown(label="Agent recommendation", elem_classes=["amd-section-card"])
            with gr.Row():
                proof = gr.Markdown(label="ROCm proof", elem_classes=["amd-section-card"])
                impact = gr.Markdown(label="Before / after", elem_classes=["amd-section-card"])
            saturation_gate = gr.Markdown(label="AMD GPU saturation gate", elem_classes=["amd-section-card"])
            cost = gr.Markdown(label="Cost estimate", elem_classes=["amd-section-card"])
            report_file = gr.File(label="Deployment report")
    chart = gr.Plot(label="Performance frontier", elem_classes=["amd-chart-panel"])
    with gr.Row():
        results = gr.Dataframe(label="Ranked configurations", interactive=False, elem_classes=["amd-table-panel"])
        scores = gr.Dataframe(label="Score breakdown", interactive=False, elem_classes=["amd-table-panel"])
    gate_matrix = gr.Dataframe(label="Production gate matrix", interactive=False, elem_classes=["amd-table-panel"])
    next_experiments = gr.Dataframe(label="Agent-planned next experiments", interactive=False, elem_classes=["amd-table-panel"])

    with gr.Tabs():
        with gr.Tab("Production Ops"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["amd-control-panel"]):
                    gr.Markdown("### Live Benchmark Runner")
                    model_task = gr.Dropdown(HF_TASKS, value="text-generation", label="Model task / category")
                    hf_query = gr.Textbox(value="Qwen2.5 Instruct", label="Search Hugging Face models")
                    hf_search_button = gr.Button("Fetch HF Models")
                    hf_status = gr.Markdown(elem_classes=["amd-section-card"])
                    hf_models = gr.Dropdown(
                        ["Qwen/Qwen2.5-7B-Instruct"],
                        value="Qwen/Qwen2.5-7B-Instruct",
                        label="Selected HF model",
                        allow_custom_value=True,
                    )
                    endpoint = gr.Textbox(value="http://localhost:8000", label="OpenAI-compatible vLLM endpoint")
                    live_model = gr.Textbox(value="", label="Manual model override")
                    workload_type = gr.Dropdown(WORKLOAD_TYPES, value="Production smoke test", label="Workload category")
                    live_workload = gr.Textbox(value="Short production smoke-test prompts", label="Workload detail")
                    live_concurrency = gr.Textbox(value="1 4 8 16", label="Concurrency values")
                    live_requests = gr.Number(value=8, precision=0, label="Requests per config")
                    auto_load_model = gr.Checkbox(value=True, label="Auto-load selected HF model on AMD droplet")
                    with gr.Accordion("AMD droplet vLLM settings", open=False):
                        ssh_target = gr.Textbox(value=DEFAULT_SSH_TARGET, label="SSH target")
                        ssh_key_path = gr.Textbox(value=DEFAULT_SSH_KEY, label="SSH key path")
                        container_name = gr.Textbox(value=DEFAULT_CONTAINER, label="ROCm container name")
                        serve_max_model_len = gr.Number(value=4096, precision=0, label="vLLM max model length")
                        serve_max_num_seqs = gr.Number(value=24, precision=0, label="vLLM max num seqs")
                        load_model_button = gr.Button("Load Selected Model On AMD Droplet")
                    live_button = gr.Button("Run Live Endpoint Benchmark", variant="primary", elem_classes=["amd-run-button"])
                    live_json = gr.File(label="Live benchmark JSON")
                with gr.Column(scale=2, elem_classes=["amd-output-panel"]):
                    live_status = gr.Markdown(elem_classes=["amd-section-card"])
                    live_results = gr.Dataframe(label="Live benchmark results", interactive=False, elem_classes=["amd-table-panel"])
                    live_regression = gr.Markdown(label="Live regression decision", elem_classes=["amd-section-card"])

            with gr.Row():
                with gr.Column(elem_classes=["amd-output-panel"]):
                    gr.Markdown("### Benchmark History And Regression")
                    with gr.Row():
                        save_history_button = gr.Button("Save Current Run To History")
                        refresh_history_button = gr.Button("Refresh History")
                        regression_button = gr.Button("Check Current Regression")
                    history_table = gr.Dataframe(label="SQLite benchmark history", interactive=False, elem_classes=["amd-table-panel"])
                    regression = gr.Markdown(label="Regression detection", elem_classes=["amd-section-card"])

        with gr.Tab("Compare And Ship"):
            with gr.Row():
                with gr.Column(elem_classes=["amd-output-panel"]):
                    gr.Markdown("### Model / Scenario Comparison")
                    compare_button = gr.Button("Compare Available Runs", variant="primary")
                    comparison_table = gr.Dataframe(label="Comparison matrix", interactive=False, elem_classes=["amd-table-panel"])
                with gr.Column(elem_classes=["amd-output-panel"]):
                    gr.Markdown("### Live ROCm Monitor")
                    monitor_refresh_button = gr.Button("Refresh ROCm Monitor")
                    monitor_live = gr.Markdown(elem_classes=["amd-section-card"])
                    monitor_graph = gr.Plot(label="ROCm SMI telemetry", elem_classes=["amd-chart-panel"])

            with gr.Row():
                with gr.Column(elem_classes=["amd-output-panel"]):
                    gr.Markdown("### Production Config Generator")
                    artifact_button = gr.Button("Generate Deployment Bundle", variant="primary")
                    artifact_summary = gr.Markdown(elem_classes=["amd-section-card"])
                    artifact_file = gr.File(label="Deployment bundle")

    run_button.click(
        dashboard,
        inputs=[
            scenario,
            benchmark_file,
            monitor_file,
            objective,
            hourly_price,
            requests_per_day,
            output_tokens_per_request,
        ],
        outputs=[
            summary_cards,
            results,
            chart,
            recommendation,
            metadata,
            proof,
            impact,
            saturation_gate,
            scores,
            gate_matrix,
            next_experiments,
            cost,
            report_file,
        ],
    )
    demo.load(
        dashboard,
        inputs=[
            scenario,
            benchmark_file,
            monitor_file,
            objective,
            hourly_price,
            requests_per_day,
            output_tokens_per_request,
        ],
        outputs=[
            summary_cards,
            results,
            chart,
            recommendation,
            metadata,
            proof,
            impact,
            saturation_gate,
            scores,
            gate_matrix,
            next_experiments,
            cost,
            report_file,
        ],
    )

    save_history_button.click(
        save_current_to_history,
        inputs=[scenario, benchmark_file, objective, hourly_price],
        outputs=[history_table, regression],
    )
    refresh_history_button.click(refresh_history, outputs=[history_table])
    regression_button.click(
        check_current_regression,
        inputs=[scenario, benchmark_file, objective, hourly_price],
        outputs=[regression],
    )
    compare_button.click(compare_models, inputs=[objective, hourly_price], outputs=[comparison_table])
    monitor_refresh_button.click(refresh_monitor_panel, inputs=[scenario, monitor_file], outputs=[monitor_live, monitor_graph])
    artifact_button.click(generate_artifacts, inputs=[scenario, benchmark_file, objective], outputs=[artifact_summary, artifact_file])
    hf_search_button.click(search_models_for_ui, inputs=[hf_query, model_task], outputs=[hf_models, live_model, hf_status])
    load_model_button.click(
        load_selected_model_on_droplet,
        inputs=[
            endpoint,
            hf_models,
            live_model,
            ssh_target,
            ssh_key_path,
            container_name,
            serve_max_model_len,
            serve_max_num_seqs,
        ],
        outputs=[live_status],
    )
    live_button.click(
        run_live_endpoint_benchmark,
        inputs=[
            endpoint,
            hf_models,
            live_model,
            workload_type,
            live_workload,
            live_concurrency,
            live_requests,
            auto_load_model,
            ssh_target,
            ssh_key_path,
            container_name,
            serve_max_model_len,
            serve_max_num_seqs,
            objective,
            hourly_price,
        ],
        outputs=[live_status, live_json, live_results, live_regression, history_table],
    )
    demo.load(refresh_history, outputs=[history_table])
    demo.load(compare_models, inputs=[objective, hourly_price], outputs=[comparison_table])
    demo.load(refresh_monitor_panel, inputs=[scenario, monitor_file], outputs=[monitor_live, monitor_graph])


if __name__ == "__main__":
    demo.launch()
