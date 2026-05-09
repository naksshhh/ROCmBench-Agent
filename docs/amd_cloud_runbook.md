# AMD Developer Cloud Runbook

Use this once your AMD Developer Cloud instance is ready. The goal is to produce one real benchmark JSON that can be uploaded into ROCmBench Agent.

## 1. Start A GPU Instance

Choose an AMD Instinct MI300X instance if available. Keep the first run short so credits are spent on proof, not idle time.

## 2. Create Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install vllm requests
```

If the image already includes ROCm, PyTorch, and vLLM, prefer the preinstalled stack.

## 3. Confirm GPU Visibility

```bash
rocm-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:", torch.version.hip)
print("cuda visible name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "not visible")
PY
```

Screenshot this for the final deck.

## 4. Serve Qwen With vLLM

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90
```

Keep this terminal open.

## 5. Run Benchmark From A Second Terminal

```bash
source .venv/bin/activate
python scripts/run_vllm_benchmark.py \
  --endpoint http://localhost:8000 \
  --model Qwen/Qwen2.5-7B-Instruct \
  --workload "Customer-support chat prompts" \
  --concurrency 1 4 8 16 \
  --requests 32 \
  --output benchmark_results/amd_qwen25_7b.json
```

## 6. Capture VRAM

During the benchmark, run:

```bash
watch -n 1 rocm-smi
```

Put observed peak VRAM into the JSON if time allows. Even one measured value makes the story stronger.

## 7. Upload Into ROCmBench Agent

Open the app, upload `benchmark_results/amd_qwen25_7b.json`, choose an optimization goal, and export the report.

## Credit Discipline

- Stop the instance immediately after collecting screenshots and JSON.
- Keep one polished real run instead of chasing many models.
- Use sample mode for UI demos and real mode for credibility.
