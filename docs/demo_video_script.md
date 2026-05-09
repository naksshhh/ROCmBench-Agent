# Demo Video Script

## 0:00 - Problem

Developers can get AMD GPU access, but choosing the right serving settings still involves guesswork. ROCmBench Agent turns benchmark runs into deployment decisions.

## 0:20 - Real AMD Run

Show the default scenario: `Real AMD MI300X run`.

Point to:

- model: `Qwen/Qwen2.5-7B-Instruct`
- backend: `ROCm + vLLM`
- hardware: `AMD Instinct MI300X`

## 0:40 - Recommendation

Click **Analyze AMD Run**.

Say:

ROCmBench recommends `concurrency-16`, which reached 2,775.85 tokens/sec, 1.471s p95 latency, and 0% errors.

## 1:05 - Proof

Show the ROCm proof panel:

- 142 ROCm samples
- 100% peak GPU utilization
- 90% VRAM allocation
- 59% memory read/write activity

## 1:25 - Business Value

Show before/after and cost estimate.

Say:

Compared with the concurrency-1 baseline, this is about 13.1x more throughput while maintaining a clean error rate. The cost estimator converts that into serving economics.

## 1:45 - Report

Open the generated report and show the vLLM launch command.

## 2:00 - Production Features

Open the Production Ops tab.

Say:

This can also run live against a vLLM endpoint, store benchmark history in SQLite, detect regressions, compare model scenarios, and generate production artifacts like a vLLM start script and docker-compose file.

Show the model picker:

Select a task like `text-generation`, search Hugging Face for Qwen or Llama, choose the model, then run the benchmark against the vLLM endpoint.

## 2:20 - Close

This is an agentic optimization workflow for AMD deployment: benchmark, rank, explain, estimate cost, and ship a deployment report.
