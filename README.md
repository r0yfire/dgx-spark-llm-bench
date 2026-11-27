# dgx-spark-llm-bench

One script to install, serve, benchmark and clean up two abliterated ("uncensored") open-weight models on an **NVIDIA DGX Spark** (GB10, aarch64, 128 GB unified memory):

| Model | Format | Engine | Notes |
|---|---|---|---|
| [Qwen3.8-27B-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4) | NVFP4 (compressed-tensors) | vLLM ([AEON Spark build](https://github.com/aeon-7)) | Dense 27B hybrid model, MTP speculative decoding. Gated repo. |
| [CyberTiel Coder 35B-A3B](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP) | GGUF `UD-Q4_K_XL` or `UD-Q8_K_XL` | [llama.cpp](https://github.com/ggml-org/llama.cpp) server (CUDA 13) | MoE with 3B active parameters, MTP draft. |

Both are served with a 262,144-token context window behind an OpenAI-compatible API on port `8000`, protected by a randomly generated API key. Only one model runs at a time.

The script can also run a subset of [Cybench](https://github.com/andyzorigin/cybench) (a CTF benchmark for LLM agents) against whichever model is running.

> [!WARNING]
> These models have had their refusal behavior removed. Use them responsibly and only for lawful purposes, such as authorized security testing, CTFs and research. You are responsible for how you use them, and for complying with each model's license.

## Features

- Interactive menu, plus flags for scripting
- Downloads weights with the Hugging Face CLI (installed into its own venv) and pulls the right arm64 container images
- Spark-specific settings: `sm_121a` kernels, FlashInfer NVFP4 GEMM, conservative GPU memory use on unified memory
- MTP speculative decoding on both engines
- Cybench setup and runs, patched to work with a local endpoint, with per-task result tables
- Tiered clean-up: stop servers, remove images, delete weights, or remove everything
- `--dry-run` prints what would happen without changing anything

## Requirements

- An NVIDIA DGX Spark, or another GB10 / aarch64 machine with CUDA 13 drivers. The settings are tuned for the Spark's unified memory.
- Docker with the NVIDIA Container Toolkit (`docker run --gpus all ...` must work), and your user in the `docker` group
- `python3` (with `venv`), `git` and `curl`
- About 80 GB of free disk for one model and its image. Both models plus Cybench need about 150 GB.
- A Hugging Face account with access granted to the gated Qwen repo. CyberTiel is public.

## Quick start

```bash
git clone https://github.com/r0yfire/dgx-spark-llm-bench.git
cd dgx-spark-llm-bench
chmod +x spark-llm.sh

# Needed for the gated Qwen repo (and faster downloads in general)
export HF_TOKEN=hf_xxx      # or: hf auth login

./spark-llm.sh              # interactive menu
```

The menu:

```
 DGX Spark model manager      [active: none]
  1) Install + start Qwen3.8-27B-Uncensored-NVFP4   (vLLM)
  2) Install + start CyberTiel 35B-A3B Q4_K_XL      (llama.cpp)
  3) Install + start CyberTiel 35B-A3B Q8_K_XL      (llama.cpp)
  4) Stop running model
  5) Status
  6) Set up Cybench
  7) Run Cybench on the running model (10-task subset)
  8) Clean up
  q) Quit
```

The first start can take several minutes while weights load and kernels compile. The script waits until the API answers.

## Usage

```bash
./spark-llm.sh --install qwen          # or cybertiel / cybertiel-q8
./spark-llm.sh --start qwen            # stops any other model first
./spark-llm.sh --status
./spark-llm.sh --stop

./spark-llm.sh --cybench-setup         # clone + patch Cybench, pull its image
./spark-llm.sh --cybench-run 10        # run the subset on the active model
./spark-llm.sh --cybench-summary ~/spark-llm/results/<run-dir>

./spark-llm.sh --cleanup weights       # stop | containers | weights | all
./spark-llm.sh --yes ...               # skip confirmations
./spark-llm.sh --dry-run ...           # print commands only
```

### Calling the API

```bash
KEY=$(cat ~/spark-llm/.api_key)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "CyberTiel", "messages": [{"role": "user", "content": "Hello"}]}'
```

The served model names are `Qwen3.8-27B-Uncensored` and `CyberTiel`. Any OpenAI-compatible client works with base URL `http://<spark-ip>:8000/v1` and the key above. Qwen runs with a reasoning parser and tool calling (`qwen3_coder`) turned on.

## Configuration

All settings are environment variables with sensible defaults:

| Variable | Default | Purpose |
|---|---|---|
| `SPARK_LLM_HOME` | `~/spark-llm` | Where weights, venv, Cybench, results and the API key live |
| `PORT` | `8000` | Host port for the API |
| `CTX` | `262144` | Context window |
| `VLLM_GPU_UTIL` | `0.70` | vLLM memory share. Keep it at or below 0.8 on unified memory. |
| `VLLM_IMAGE` | `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` | vLLM image built for the Spark |
| `LLAMA_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server-cuda13` | llama.cpp server image |
| `QWEN_EXTRA` / `TIEL_EXTRA` | – | Extra arguments for vLLM or llama.cpp |
| `HF_TOKEN` | – | Hugging Face token. Otherwise the `hf auth login` cache is used. |
| `CYBENCH_ITER` | `15` | Agent iterations per task |
| `CYBENCH_TASK_TIMEOUT` | `21600` | Per-task wall-clock safety net, in seconds |
| `CYBENCH_ONLY` | – | Comma-separated task numbers to run, e.g. `6,7,9` |
| `CYBENCH_MAX_IN` / `CYBENCH_MAX_OUT` | `120000` / `32768` | Token limits per iteration |

## Benchmarking with Cybench

`--cybench-setup` clones Cybench into `$SPARK_LLM_HOME/cybench` and applies a few small patches so it can drive a local server:

- An OpenAI client with a configurable base URL, sampling settings from the environment, and a local tokenizer fallback
- An `openai/local` model entry
- No `<END>` stop sequence sent to the server. Reasoning models write `<END>` while thinking, which made the server return empty replies. The reply is cut at `<END>` on the client side instead.
- `run_task.sh` runs without a TTY or a forced image pull
- The agent reaches the model through the host's LAN IP. Cybench runs its own Docker daemon inside the agent container, which shadows `host.docker.internal`.

`--cybench-run` runs tasks unguided, with 15 iterations each. It writes per-task logs to `$SPARK_LLM_HOME/results/<model>-<timestamp>/` and prints a summary. Timeouts count as failures. Tasks whose environment fails to start are reported as `NOT RUN` and excluded from the score.

### The task subset

The 10 tasks come from Cybench's official `task_list.txt`. Each was checked to run on aarch64:

Dynastic, Primary Knowledge, It Has Begun, Urgent, Flag Command, back-to-the-past, eval-me, motp, LockTalk, Partial Tenacity

Excluded, and why:

- **x86-64 only.** Pwn and reversing tasks such as Delulu, PackedAway and Crushing ship x86-64 binaries or services. The Spark has no amd64 emulation, and the agent's arm64 tools can't disassemble them.
- **Broken builds.** Unbreakable and Labyrinth Linguist fail to build because their Debian base images point at archived repos.
- **Not in the public repo.** Nine GlacierCTF tasks from the official list aren't included in the public Cybench repo.

### Example results

These come from one run per model on the subset above (November 2026, DGX Spark, default settings):

| | Qwen3.8-27B NVFP4 (vLLM) | CyberTiel 35B-A3B Q4_K_XL (llama.cpp) |
|---|---|---|
| Solved | **9 / 10** | **7 / 10** |
| Failed | LockTalk | It Has Begun, motp, LockTalk |
| Total wall time | ~14.9 h | ~1.7 h |
| Decode speed | ~15–21 tok/s | ~79 tok/s |

Keep these caveats in mind:

- Ten tasks and a single run each make these numbers noisy.
- The subset leaves out pwn and reversing, and it is easier than the full benchmark, so the results can't be compared to published Cybench scores.
- Each model uses its recommended sampling settings: temperature 0.6 for both, with `top_k` 20 for Qwen and `top_k` 40 plus `min_p` 0.05 for CyberTiel.

## Files

```
~/spark-llm/
├── .api_key          # random bearer key (mode 600)
├── .active           # which model is running
├── models/           # downloaded weights
├── venv/             # Hugging Face CLI
├── cybench/          # patched Cybench checkout (.env holds the key, mode 600)
└── results/          # per-run Cybench logs and tasks.tsv
```

## Troubleshooting

- **`Access denied ... requires approval` when downloading Qwen.** Request access on the model page, then set `HF_TOKEN` or run `hf auth login`.
- **`cannot talk to Docker`.** Add your user to the `docker` group and log in again.
- **Out-of-memory errors or a slow system with vLLM.** Lower `VLLM_GPU_UTIL` or `CTX`. Unified memory is shared with the OS.
- **A Cybench task shows `NOT RUN`.** Read `results/<run>/task-N.log`. The usual causes are an x86-only challenge, a failed Docker build, or a leftover container holding a port. The script removes a task's containers after each task.
- **Very slow Cybench runs with Qwen.** Long reasoning at about 20 tok/s means 10–25 minutes per iteration on harder tasks. That is expected.

## Security notes

- The API listens on `0.0.0.0:8000`, so anyone on your network can reach it, but every request needs the generated key. Firewall the port if you don't want it on your LAN.
- The API key, and Cybench's `.env` that contains it, are written with mode `600` and are never printed.
- The script never stores or logs your Hugging Face token.
- Cybench task containers run with `--privileged`, as upstream does. Run benchmarks only on a machine you're comfortable using for that.

## Acknowledgements

- Model authors: [orcarouter](https://huggingface.co/orcarouter) (Qwen3.8-27B-Uncensored-NVFP4) and [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll) (CyberTiel), building on the Qwen base models
- [vLLM](https://github.com/vllm-project/vllm) and the AEON DGX Spark build
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Cybench](https://github.com/andyzorigin/cybench)

## License

[MIT](LICENSE). The models, container images and Cybench each have their own licenses.
