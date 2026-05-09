# Submission Checklist

- Project title: ROCmBench Agent
- Short description: AI deployment copilot that benchmarks and optimizes open-source LLM serving on AMD GPUs.
- Track: AI Agents & Agentic Workflows
- Special rewards: Qwen, Hugging Face Space, Build in Public
- GitHub repository: public, MIT license
- Demo application: Hugging Face Space
- Video: 2 minutes, show upload -> chart -> recommendation -> report
- Slides: 5 slides
- Required proof: AMD Developer Cloud screenshot, `rocm-smi`, benchmark JSON, generated report
- Real benchmark JSON: `benchmark_results/amd_qwen25_7b.json`
- ROCm monitor log: `benchmark_results/amd_qwen25_7b_rocm_monitor.log`
- Real generated report: `reports/rocmbench_Qwen_Qwen2.5-7B-Instruct_latest.md`
- Production features: live benchmark runner, SQLite history, regression detection, model comparison, config bundle generator, ROCm monitor refresh, deployment decision report

## Slide Outline

1. Problem: AMD GPU deployment still requires guesswork.
2. Product: ROCmBench Agent workflow.
3. Demo: performance frontier and `concurrency-16` recommendation.
4. AMD proof: 2,775.85 tok/s, 1.471s p95 latency, 0% errors, 100% peak GPU utilization.
5. Business value: lower cost, faster deployment, reusable readiness checks.

## Final Submission Description

ROCmBench Agent is an AI deployment copilot for open-source LLM workloads on AMD GPUs. Developers upload or generate benchmark results from AMD Developer Cloud, and the system ranks vLLM serving configurations across latency, throughput, memory, and reliability. It then produces an AMD readiness score, a recommended launch command, and a deployment report suitable for engineering and business stakeholders.

The project uses AMD Developer Cloud, ROCm, vLLM, and Qwen to move beyond a simple chatbot into an agentic optimization workflow: profile the workload, benchmark configurations, reason over tradeoffs, and produce an actionable deployment plan.
