# dgx-spark-llm-bench

One script to install, serve, benchmark and clean up abliterated ("uncensored") open-weight models on an **NVIDIA DGX Spark** (GB10, aarch64, 128 GB unified memory), on whichever inference engine runs each model fastest there:

| Engine | What it runs here | How it runs |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) ([AEON Spark build](https://github.com/aeon-7)) | NVFP4 checkpoints | Docker |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) server (CUDA 13) | GGUF | Docker |
| [PolyStrata](https://github.com/simplexai-labs/PolyStrata) (`strata-poly`) | GGUF mixture-of-experts models | native, built from [our fork](https://github.com/r0yfire/PolyStrata/tree/spark-aarch64) |
| [TensorFold](https://github.com/ashhart/TensorFold) | MLX 4-bit checkpoints, with a DFlash2 drafter | native release |

| Model | Engines (default first) | Notes |
|---|---|---|
| `qwen27b-abl`: [Qwen3.8-27B Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4) | `tensorfold` ([MLX 4-bit](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX)), `vllm` ([NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4)) | Dense 27B hybrid model. TensorFold drafts with [z-lab's DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2); vLLM with the model's MTP head. The NVFP4 repo is gated. |
| `cybertiel`, `cybertiel-q8`: [CyberTiel Coder 35B-A3B](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP) | `polystrata`, `llamacpp` | MoE with 3B active parameters, GGUF `UD-Q4_K_XL` or `UD-Q8_K_XL`, MTP draft head in the file. |

Every model is served with a 262,144-token context window behind an OpenAI-compatible API on port `8000`, protected by a randomly generated API key. Only one model runs at a time.

The script also measures tokens per second (prefill and decode) on the running model, and can run a subset of [Cybench](https://github.com/andyzorigin/cybench) (a CTF benchmark for LLM agents) against it.

> [!WARNING]
> These models have had their refusal behavior removed. Use them responsibly and only for lawful purposes, such as authorized security testing, CTFs and research. You are responsible for how you use them, and for complying with each model's license.

## Results

Measured on one DGX Spark (November 2026) with `speedtest.py` from this repo: every engine gets the same prompts through the same streaming client, temperature 0, 512-token answers, thinking off, 3 runs per cell, medians. Prefill is prompt tokens divided by time to first token; decode is the answer's tokens divided by the time from the first to the last. The needle test hides a code at one third of a ~208,600-token prompt and asks for it back.

| Model | Engine | Decode tok/s (4K / 16K / 28K prompt) | Prefill tok/s (4K / 16K / 28K) | ~208K needle |
|---|---|---|---|---|
| Qwen3.8-27B Uncensored | vLLM, NVFP4, MTP 2 drafts (previous default) | 18.9 / 18.0 / 16.9 | 1,887 / 1,971 / 1,808 | found, first token after 259 s |
| Qwen3.8-27B Uncensored | vLLM, NVFP4, MTP 4 drafts | 20.7 / 20.2 / 19.4 | 1,959 / 1,984 / 1,783 | found, 262 s |
| Qwen3.8-27B Uncensored | **TensorFold**, MLX 4-bit + DFlash2 | **59.1 / 58.4 / 54.2** | 1,414 / 1,349 / 1,316 | found, **253 s** |
| CyberTiel 35B-A3B Q4_K_XL | llama.cpp, MTP | 84.5 / 80.2 / 74.4 | 2,300 / 2,281 / 2,172 | found, 144 s |
| CyberTiel 35B-A3B Q4_K_XL | **PolyStrata**, MTP | **97.2 / 87.6 / 85.9** | **3,648 / 3,583 / 3,491** | found, **100 s** |

What these show:

- **Qwen3.8-27B:** TensorFold decodes about 3 times as fast as vLLM. vLLM reads prompts faster below ~32K, and at ~208K the two are level. vLLM's best setting is 4 MTP drafts: 6 drafts and a 16K prefill batch measured no better (prefill at 16K and 32K got slower with the larger batch). TensorFold with an FP8 KV cache decoded slower (50–53 tok/s) than with the default cache.
- **CyberTiel:** PolyStrata decodes 12–15% faster than llama.cpp and reads prompts about 60% faster, on the same GGUF file. Greedy answers to 5 fixed prompts (`--parity`) were identical on 3; the other 2 parted at a near-tie token after 65 and 160 characters, both still correct. PolyStrata's own docs note its fused kernels round slightly differently from llama.cpp's.
- The two Qwen3.8-27B engines read different 4-bit formats of the same OrcaRouter weights (MLX affine 4-bit for TensorFold, NVFP4 for vLLM), so their answers are not expected to match token for token.
- Single runs on one machine; expect a few percent of variation between sessions.

## Features

- Interactive menu, plus flags for scripting
- One registry row per model and engine: each engine downloads the format it reads
- Downloads weights with the Hugging Face CLI (installed into its own venv), pulls the arm64 container images, builds PolyStrata's `strata-poly` for `sm_121`, and installs TensorFold's release after checking its SHA-256
- Spark-specific settings: `sm_121a` kernels, FlashInfer NVFP4 GEMM, conservative GPU memory use on unified memory
- Speculative decoding on every engine (MTP, or DFlash2 on TensorFold)
- `--speed` and `--parity` benchmarks of the running model
- Cybench setup and runs, patched to work with a local endpoint, with per-task result tables
- Tiered clean-up: stop servers, remove images, delete weights, or remove everything
- `--dry-run` prints what would happen without changing anything

## Requirements

- An NVIDIA DGX Spark, or another GB10 / aarch64 machine with CUDA 13 drivers (580 or newer). The settings are tuned for the Spark's unified memory.
- Docker with the NVIDIA Container Toolkit (`docker run --gpus all ...` must work), and your user in the `docker` group
- `python3` (with `venv`), `git`, `curl` and `sha256sum`
- For PolyStrata: the CUDA 13 toolkit (`/usr/local/cuda/bin/nvcc`, which DGX OS ships) and `cmake`. The build takes about 10 minutes.
- About 80 GB of free disk per model and engine.
- A Hugging Face account with access granted to gated repos (the Qwen NVFP4 repo). The others are public.

## Quick start

```bash
git clone https://github.com/r0yfire/dgx-spark-llm-bench.git
cd dgx-spark-llm-bench
chmod +x spark-llm.sh

# Needed for gated repos (and faster downloads in general)
export HF_TOKEN=hf_xxx      # or: hf auth login

./spark-llm.sh              # interactive menu
```

The menu lists every model and engine pair:

```
 DGX Spark model manager      [active: none]
   1) Qwen3.8-27B Uncensored, MLX 4-bit + DFlash2      tensorfold
   2) Qwen3.8-27B Uncensored, NVFP4 + MTP              vllm
   3) CyberTiel 35B-A3B UD-Q4_K_XL                     polystrata
   ...
   s) Stop running model       t) Status          p) Speed test (tok/s)
   c) Set up Cybench           r) Run Cybench on the running model (10-task subset)
   x) Clean up                 q) Quit
```

The first start can take several minutes while weights load and kernels compile. The script waits until the API answers.

## Usage

```bash
./spark-llm.sh --list                              # models and the engines each one runs on
./spark-llm.sh --install qwen27b-abl               # the default (fastest) engine's files
./spark-llm.sh --start qwen27b-abl                 # stops any other model first
./spark-llm.sh --start qwen27b-abl --engine vllm   # another engine for the same model
./spark-llm.sh --status
./spark-llm.sh --stop

./spark-llm.sh --speed                             # tokens/s of the running model (short, 4K, 16K, 32K prompts)
SPEED_NEEDLE=236000 ./spark-llm.sh --speed 4k      # one prompt size, plus a ~208K-token needle test
./spark-llm.sh --parity                            # greedy answers to fixed prompts, to compare engines

./spark-llm.sh --cybench-setup                     # clone + patch Cybench, pull its image
./spark-llm.sh --cybench-run 10                    # run the subset on the active model
./spark-llm.sh --cybench-summary ~/spark-llm/results/<run-dir>

./spark-llm.sh --cleanup weights                   # stop | containers | weights | all
./spark-llm.sh --yes ...                           # skip confirmations
./spark-llm.sh --dry-run ...                       # print commands only
```

`qwen` and `tiel` still work as aliases for `qwen27b-abl` and `cybertiel`.

### Calling the API

```bash
KEY=$(cat ~/spark-llm/.api_key)
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model": "CyberTiel", "messages": [{"role": "user", "content": "Hello"}]}'
```

The served model names are `Qwen3.8-27B-Uncensored` and `CyberTiel`, whichever engine runs them. Any OpenAI-compatible client works with base URL `http://<spark-ip>:8000/v1` and the key above. On vLLM, Qwen runs with a reasoning parser and tool calling (`qwen3_coder`) turned on. To turn thinking off for a request, send `"chat_template_kwargs": {"enable_thinking": false}`; every engine here honors it.

## Engines

### PolyStrata, from a fork

PolyStrata runs mixture-of-experts GGUF models with its own CUDA engine, `strata-poly`. Upstream ships no aarch64 build of it, and it did not compile on aarch64: two files included the x86-only `<immintrin.h>` for a spin-wait hint. The [`spark-aarch64` branch of our fork](https://github.com/r0yfire/PolyStrata/tree/spark-aarch64) uses PolyStrata's own portable `cpu_relax.hpp` there instead (the same instruction on x86, `yield` on Arm). The script clones that branch at a pinned commit (`PS_REPO`, `PS_REF`), builds `strata-poly` for `sm_121`, exports the model's tokenizer and serves it with PolyStrata's server. With all experts resident on the Spark, PolyStrata's advantage over llama.cpp comes from its kernels, not from expert offloading.

PolyStrata's Spark-supported Qwen3.8-Flash-Next path uses a separate engine and is not wired into this script yet.

### TensorFold

TensorFold's native release (CUDA kernels embedded in the binary) serves Qwen3.8-27B from an MLX affine 4-bit checkpoint, drafting with DFlash2. The script installs release `TF_VERSION` after checking the published SHA-256, and passes the API key through `--api-key-file`. The native release does not accept `--prefill-fp8` yet (only TensorFold's Python engine does), which is why its prefill trails vLLM on shorter prompts.

### vLLM

vLLM drafts with the model's MTP head, 4 tokens by default (`VLLM_SPEC`). The AEON image also contains vLLM's DFlash support, but its DFlash2 drafter failed to start there (`FlashInferImpl.__init__() got an unexpected keyword argument 'use_mm_prefix'`), so it is not the default.

## Configuration

All settings are environment variables with sensible defaults:

| Variable | Default | Purpose |
|---|---|---|
| `SPARK_LLM_HOME` | `~/spark-llm` | Where weights, engines, venv, Cybench, results and the API key live |
| `PORT` | `8000` | Host port for the API |
| `CTX` | `262144` | Context window |
| `VLLM_GPU_UTIL` | `0.70` | vLLM memory share. Keep it at or below 0.8 on unified memory. |
| `VLLM_SPEC` | `{"method":"mtp","num_speculative_tokens":4}` | vLLM's `--speculative-config` |
| `VLLM_IMAGE` | `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` | vLLM image built for the Spark |
| `LLAMA_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server-cuda13` | llama.cpp server image |
| `PS_REPO` / `PS_REF` | our fork / a pinned commit | PolyStrata source to build |
| `TF_VERSION` | `1.3.6` | TensorFold release |
| `DRAFTER_27B_REPO` | `z-lab/Qwen3.8-27B-DFlash2` | DFlash2 drafter for Qwen3.8-27B |
| `VLLM_EXTRA` / `LLAMA_EXTRA` / `TF_EXTRA` | – | Extra arguments for vLLM, llama.cpp or TensorFold (`QWEN_EXTRA` / `TIEL_EXTRA` still work) |
| `SPEED_RUNS` / `SPEED_NEEDLE` | `3` / `0` | Runs per prompt size; needle prompt size in tokens (0 skips it) |
| `HF_TOKEN` | – | Hugging Face token. Otherwise the `hf auth login` cache is used. |
| `CYBENCH_ITER` | `15` | Agent iterations per task |
| `CYBENCH_TASK_TIMEOUT` | `21600` | Per-task wall-clock safety net, in seconds |
| `CYBENCH_ONLY` | – | Comma-separated task numbers to run, e.g. `6,7,9` |
| `CYBENCH_MAX_IN` / `CYBENCH_MAX_OUT` | `120000` / `32768` | Token limits per iteration |

## Speed test

`speedtest.py` (standard-library Python, no dependencies) drives any OpenAI-compatible server:

- Each prompt is real source code (the Python standard library on the machine, read in a fixed order) and starts with a random nonce, so no prefix or conversation cache can skip the prefill.
- Token counts come from the server's own `usage` (`stream_options.include_usage`).
- `--needle N` checks long context; `--parity` saves greedy answers to fixed prompts; `--table FILES...` prints a markdown table of result files.

`--speed` writes results to `$SPARK_LLM_HOME/speed/<model>-<engine>-<timestamp>.json`.

## Benchmarking with Cybench

`--cybench-setup` clones Cybench into `$SPARK_LLM_HOME/cybench` and applies a few small patches so it can drive a local server:

- An OpenAI client with a configurable base URL, sampling settings from the environment, and a local tokenizer fallback
- An `openai/local` model entry
- No `<END>` stop sequence sent to the server. Reasoning models write `<END>` while thinking, which made the server return empty replies. The reply is cut at `<END>` on the client side instead.
- `run_task.sh` runs without a TTY or a forced image pull
- The agent reaches the model through the host's LAN IP. Cybench runs its own Docker daemon inside the agent container, which shadows `host.docker.internal`.

`--cybench-run` runs tasks unguided, with 15 iterations each. It writes per-task logs to `$SPARK_LLM_HOME/results/<model>-<engine>-<timestamp>/` and prints a summary. Timeouts count as failures. Tasks whose environment fails to start are reported as `NOT RUN` and excluded from the score.

### The task subset

The 10 tasks come from Cybench's official `task_list.txt`. Each was checked to run on aarch64:

Dynastic, Primary Knowledge, It Has Begun, Urgent, Flag Command, back-to-the-past, eval-me, motp, LockTalk, Partial Tenacity

Excluded, and why:

- **x86-64 only.** Pwn and reversing tasks such as Delulu, PackedAway and Crushing ship x86-64 binaries or services. The Spark has no amd64 emulation, and the agent's arm64 tools can't disassemble them.
- **Broken builds.** Unbreakable and Labyrinth Linguist fail to build because their Debian base images point at archived repos.
- **Not in the public repo.** Nine GlacierCTF tasks from the official list aren't included in the public Cybench repo.

### Example results

These come from one run per model on the subset above (November 2026, DGX Spark), with the engines this script used at the time (vLLM with 2 MTP drafts, and llama.cpp):

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
├── .api_key            # random bearer key (mode 600)
├── .active             # which model and engine are running
├── models/             # downloaded weights, one folder per model and format
├── polystrata/         # PolyStrata checkout and its strata-poly build
├── polystrata-data/    # tokenizer packs and run configs
├── tensorfold/         # TensorFold release
├── venv/               # Hugging Face CLI
├── speed/              # speed and parity results (JSON)
├── logs/               # native engines' server logs
├── cybench/            # patched Cybench checkout (.env holds the key, mode 600)
└── results/            # per-run Cybench logs and tasks.tsv
```

## Troubleshooting

- **`Access denied ... requires approval` when downloading.** Request access on the model page, then set `HF_TOKEN` or run `hf auth login`.
- **`cannot talk to Docker`.** Add your user to the `docker` group and log in again.
- **`port 8000 is still in use after stopping everything`.** Something outside the script holds the port: `ss -ltnp '( sport = :8000 )'` shows what.
- **PolyStrata or TensorFold exits at start.** Read `~/spark-llm/logs/polystrata-server.log` or `~/spark-llm/logs/tensorfold-<model>.log`. A failed PolyStrata build logs to `~/spark-llm/logs/polystrata-build.log`.
- **Out-of-memory errors or a slow system.** Lower `CTX` (or `VLLM_GPU_UTIL` for vLLM). Unified memory is shared with the OS.
- **A Cybench task shows `NOT RUN`.** Read `results/<run>/task-N.log`. The usual causes are an x86-only challenge, a failed Docker build, or a leftover container holding a port. The script removes a task's containers after each task.

## Security notes

- The API listens on `0.0.0.0:8000`, so anyone on your network can reach it, but every request needs the generated key. Firewall the port if you don't want it on your LAN.
- The API key, and Cybench's `.env` that contains it, are written with mode `600` and are never printed. Native engines get it from that file (TensorFold) or through the environment (PolyStrata), not on the command line.
- The script never stores or logs your Hugging Face token.
- PolyStrata is built from a pinned commit of our fork, and TensorFold's release is checked against its published SHA-256 before use.
- Cybench task containers run with `--privileged`, as upstream does. Run benchmarks only on a machine you're comfortable using for that.

## Acknowledgements

- Model authors: [orcarouter](https://huggingface.co/orcarouter) (Qwen3.8-27B Uncensored) and [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll) (CyberTiel), building on the Qwen base models; [z-lab](https://huggingface.co/z-lab) (DFlash2 drafter)
- [vLLM](https://github.com/vllm-project/vllm) and the AEON DGX Spark build
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [PolyStrata](https://github.com/simplexai-labs/PolyStrata) and [Strata](https://github.com/Niko1221/Strata)
- [TensorFold](https://github.com/ashhart/TensorFold)
- [Cybench](https://github.com/andyzorigin/cybench)

## License

[MIT](LICENSE). The models, engines, container images and Cybench each have their own licenses.
