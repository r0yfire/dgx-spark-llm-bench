# dgx-spark-llm-bench

One script to install, serve, benchmark and clean up abliterated ("uncensored") open-weight models on an **NVIDIA DGX Spark** (GB10, aarch64, 128 GB unified memory), on whichever inference engine runs each model fastest there:

| Engine | What it runs here | How it runs |
|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) ([AEON Spark build](https://github.com/aeon-7)) | NVFP4 checkpoints | Docker |
| vLLM v0.33 ([blazux's Flash-Next recipe](https://github.com/blazux/qwen3.8-Flash-DGX)) | Qwen3.8-Flash-Next ModelOpt NVFP4 | Docker: an image built here from the recipe at a pinned commit, started by the recipe's own `serve.sh` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) server (CUDA 13) | GGUF | Docker |
| [PolyStrata](https://github.com/simplexai-labs/PolyStrata) | GGUF mixture-of-experts models | native: `strata-poly` built from [our fork](https://github.com/r0yfire/PolyStrata/tree/spark-aarch64) (CyberTiel), the Strata engine's signed release (Qwen3.8-Flash-Next) |
| [Strata](https://github.com/Niko1221/Strata) v0.2.13 | DeepSeek-V4.1-Flash GGUF (ISTA-DASLab's GSQ-RCO sets), with the DSpark drafter | native: the release's checkout and its signed DeepSeek engine package (`strata-dsv41`) |
| [TensorFold](https://github.com/ashhart/TensorFold) | MLX 4-bit checkpoints: Qwen3.8-27B with a DFlash2 drafter, Qwen3.8-Flash-Next with its MTP head | native release |
| [veloGB10](https://github.com/sf-stav/veloGB10) | EXL3 checkpoints with their MTP head (GB10 only) | native release, behind `auth-proxy.py` from this repo (the engine has no API-key option) |

| Model | Engines (default first) | Notes |
|---|---|---|
| `qwen27b-abl`: [Qwen3.8-27B Uncensored](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4) | `tensorfold` ([MLX 4-bit](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-MLX)), `vllm` ([NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4)) | Dense 27B hybrid model. TensorFold drafts with [z-lab's DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2); vLLM with the model's MTP head. The NVFP4 repo is gated. |
| `qwen27b`: [Qwen3.8-27B](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4) (stock) | `tensorfold` ([MLX 4-bit](https://huggingface.co/mlx-community/Qwen3.8-27B-4bit)), `vllm` ([NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4)) | The original model, as the baseline for the abliterated one: the same two formats (MLX affine 4-bit; NVFP4 + FP8 with the same tensors in each: NVFP4 feed-forward weights in layers 0-55, FP8 in layers 56-63 and in every attention projection, and the output layer and MTP head unquantized) and the same drafters, at the revisions measured. On vLLM the script mounts a corrected `config.json` (see [vLLM](#vllm)). |
| `cybertiel`, `cybertiel-q8`: [CyberTiel Coder 35B-A3B](https://huggingface.co/peculiar-ragdoll/Cyber-Tiel-Coder-35B-A3B-GGUF-MTP) | `polystrata`, `llamacpp` | MoE with 3B active parameters, GGUF `UD-Q4_K_XL` or `UD-Q8_K_XL`, MTP draft head in the file. |
| `flashnext`: [Qwen3.8-Flash-Next](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (stock) | `polystrata`, `llamacpp` | The original model, as the baseline for the abliterated one: ISTA-DASLab's GSQ-RCO `IQ3_S` GGUF (two shards, about 84 GB) at the revision PolyStrata pins. PolyStrata drafts with DFlash2; llama.cpp runs without a drafter. PolyStrata can add its experimental refusal-direction projection (`PS_REFUSAL_PROJECTION=on`). |
| `flashnext-abl`: [Qwen3.8-Flash-Next Uncensored](https://huggingface.co/orcarouter/Qwen3.8-Flash-Next-Uncensored-GGUF) | `polystrata`, `llamacpp`, `tensorfold` ([MLX 4-bit + MTP](https://huggingface.co/yikoudazhong/Qwen3.8-Flash-Next-Uncensored-TF-4bit-MTP)), `velogb10` ([EXL3 3.05 bpw](https://huggingface.co/Lygodactylus/Qwen3.8-Flash-Next-Uncensored-exl3-3bpw)), `vllm-flash` ([ModelOpt NVFP4](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4)) | OrcaRouter's abliteration in four formats. `polystrata` and `llamacpp` read OrcaRouter's `IQ3_XXS` GGUF (two shards, 85.2 GB; the repo is gated): PolyStrata drafts with the same DFlash2 drafter as the stock model, llama.cpp with OrcaRouter's separate MTP draft model (`LLAMA_ABL_MTP=off` turns it off). `tensorfold` reads yikoudazhong's MLX affine 4-bit build (group size 32, 107 GB, in the layout of TensorFold's own Flash-Next checkpoint), `velogb10` Lygodactylus's EXL3 build (3.05 bits per weight, 68 GB), and `vllm-flash` jpezzulli's ModelOpt NVFP4 build (routed experts in NVFP4, the other layers BF16, the n-gram table FP8; 135 GB, in RadixArk's layout); all three draft with the model's MTP head, and all three repos are public and pinned to the revisions measured. The four formats are different quantizations, so their answers are not expected to match token for token. |
| `deepseek41`: [DeepSeek-V4.1-Flash-1017](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) (stock) | `strata` | The original model, as the baseline for the abliterated one: ISTA-DASLab's GSQ-RCO `IQ2_XXS` set (2.00 bits per weight, four shards, 138 GB) at the revision Strata v0.2.13 pins, plus three files both sets read: ISTA's Engram table (110.6 GB), the DSpark drafter (8 GB) and a 5.8 MB metadata shard from [Unsloth's repo](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF). Served with its native 1,048,576-token context. |
| `deepseek41-abl`: [DeepSeek-V4.1-Flash-1017 Uncensored](https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF) | `strata` | AtomicChat's abliterated `IQ2_XXS` set, in the same layout. Its shards 1-3 are ISTA's, byte for byte; only shard 4 (20.5 GB) differs, so the second set to be installed downloads only that shard and links the other three. Served with the native 1,048,576-token context. |

Every model is served with a 262,144-token context window (DeepSeek-V4.1-Flash with its native 1,048,576) behind an OpenAI-compatible API on port `8000`, protected by a randomly generated API key that every inference route requires (the script checks this at each start). Only one model runs at a time.

The script also measures tokens per second (prefill and decode) on the running model, and can run a subset of [Cybench](https://github.com/andyzorigin/cybench) (a CTF benchmark for LLM agents) against it.

> [!WARNING]
> These models have had their refusal behavior removed. The exceptions are the stock models, `qwen27b`, `deepseek41` and `flashnext`; the latter unless you turn on PolyStrata's refusal-direction projection for it (its package says the model then declines far fewer requests). Use them responsibly and only for lawful purposes, such as authorized security testing, CTFs and research. You are responsible for how you use them, and for complying with each model's license.

## Results

Measured on one DGX Spark (November 2026) with `speedtest.py` from this repo: every engine gets the same prompts through the same streaming client, temperature 0, 512-token answers, thinking off, 3 runs per cell, medians. Prefill is prompt tokens divided by time to first token; decode is the answer's tokens minus one (`completion_tokens - 1`, as reported by the server) divided by the time from the first streamed token to the last. The prompt cases are named by nominal size (`4k`, `16k`, `32k`); the prompts the engines actually counted were about 3,764-4,050 tokens (4k), 13,951-14,430 (16k) and 27,440-27,950 (32k), depending on the model's tokenizer. The needle test hides a code at one third of a long prompt and asks for it back: ~208,600 tokens with Qwen's and CyberTiel's tokenizers, about 204,930 with DeepSeek's.

| Model | Engine | Decode tok/s (4k / 16k / 32k case) | Prefill tok/s (4k / 16k / 32k case) | ~208K needle |
|---|---|---|---|---|
| Qwen3.8-27B Uncensored | vLLM, NVFP4, MTP 2 drafts (previous default) | 18.9 / 18.0 / 16.9 | 1,887 / 1,971 / 1,808 | found, first token after 259 s |
| Qwen3.8-27B Uncensored | vLLM, NVFP4, MTP 4 drafts | 20.7 / 20.2 / 19.4 | 1,959 / 1,984 / 1,783 | found, 262 s |
| Qwen3.8-27B Uncensored | **TensorFold**, MLX 4-bit + DFlash2 | **59.1 / 58.4 / 54.2** | 1,414 / 1,349 / 1,316 | found, **253 s** |
| Qwen3.8-27B (stock) | vLLM, NVFP4, MTP 4 drafts | 21.2 / 19.9 / 18.7 | 1,960 / 1,993 / 1,828 | found, 261 s |
| Qwen3.8-27B (stock) | **TensorFold**, MLX 4-bit + DFlash2 | **59.8 / 53.8 / 52.3** | 1,421 / 1,353 / 1,306 | found, **253 s** |
| CyberTiel 35B-A3B Q4_K_XL | llama.cpp, MTP | 84.5 / 80.2 / 74.4 | 2,300 / 2,281 / 2,172 | found, 144 s |
| CyberTiel 35B-A3B Q4_K_XL | **PolyStrata**, MTP | **97.2 / 87.6 / 85.9** | **3,648 / 3,583 / 3,491** | found, **100 s** |
| Qwen3.8-Flash-Next (stock) IQ3_S | llama.cpp, no drafter | 31.2 / 30.8 / 30.0 | 858 / 848 / 817 | found, 276 s |
| Qwen3.8-Flash-Next (stock) IQ3_S | **PolyStrata**, DFlash2 | **63.3 / 61.7 / 62.3** | **1,477 / 1,857 / 1,992** | found, **111 s** |
| Qwen3.8-Flash-Next (stock) IQ3_S | PolyStrata, DFlash2 + refusal projection | 60.2 / 59.9 / 60.4 | 1,431 / 1,787 / 1,904 | found, 116 s |
| Qwen3.8-Flash-Next Uncensored IQ3_XXS | llama.cpp, no drafter (`LLAMA_ABL_MTP=off`) | 41.0 / 40.2 / 39.3 | 841 / 837 / 800 | found, 281 s |
| Qwen3.8-Flash-Next Uncensored IQ3_XXS | llama.cpp, MTP draft model | 61.3 / 59.8 / 58.2 | 784 / 787 / 753 | found, 308 s |
| Qwen3.8-Flash-Next Uncensored IQ3_XXS | PolyStrata, DFlash2 | 67.4 / 65.1 / 64.6 | 1,573 / 1,931 / 2,063 | found, 108 s |
| Qwen3.8-Flash-Next Uncensored MLX 4-bit | TensorFold, MTP | 70.5 / 68.8 / 66.2 | 1,962 / 2,180 / 2,188 | found, 174 s |
| Qwen3.8-Flash-Next Uncensored EXL3 3.05 bpw | **veloGB10**, MTP | **86.0 / 83.3 / 83.5** | 1,783 / 1,724 / 1,543 | found, 162 s |
| Qwen3.8-Flash-Next Uncensored ModelOpt NVFP4 | vLLM v0.33 (blazux's recipe), as published, MTP 3 drafts | 37.8 / 38.9 / 38.0 | 2,647 / 3,108 / 3,049 | found, 76.8 s |
| Qwen3.8-Flash-Next Uncensored ModelOpt NVFP4 | **vLLM v0.33** (blazux's recipe), fp8 side layers, MTP 3 drafts | 46.8 / 44.9 / 45.0 | **2,796 / 3,332 / 3,273** | found, **73.6 s** |
| DeepSeek-V4.1-Flash-1017 (stock) IQ2_XXS | Strata v0.2.13, DSpark | 27.3 / 26.4 / 24.9 | 269 / 333 / 445 | found, 450 s |
| DeepSeek-V4.1-Flash-1017 Uncensored IQ2_XXS | Strata v0.2.13, DSpark | 24.6 / 28.4 / 27.5 | 272 / 433 / 489 | found, 388 s |

What these show:

- **Qwen3.8-27B:** TensorFold decodes about 3 times as fast as vLLM. vLLM reads prompts faster in the 4k to 32k cases, and at ~208K the two are level. vLLM's best setting is 4 MTP drafts: 6 drafts and a 16K prefill batch measured no better (prefill in the 16k and 32k cases got slower with the larger batch). TensorFold with an FP8 KV cache decoded slower (50–53 tok/s) than with the default cache. The stock model measured like the abliterated one on both engines: within 4% in every cell except TensorFold's decode in the 16k case, 8% lower for the stock model.
- **CyberTiel:** PolyStrata decodes 9-15% faster than llama.cpp (15% in the 4k case, 9% at 16k, 15% at 32k) and reads prompts about 60% faster, on the same GGUF file. Greedy answers to 5 fixed prompts (`--parity`) were identical on 3; the other 2 parted at a near-tie token after 65 and 160 characters, both still correct. PolyStrata's own docs note its fused kernels round slightly differently from llama.cpp's.
- **Qwen3.8-Flash-Next on the GGUF files:** PolyStrata is the faster engine for both models. It decodes about twice as fast as llama.cpp without a drafter (2.0 times for the stock model, 1.6 times for the abliterated one) and 9-11% faster than llama.cpp with OrcaRouter's MTP draft model. It reads prompts 1.7 to 2.7 times as fast (the gap grows with the prompt) and answers the ~208K needle in under 2 minutes, against 4.6 to 5.1 minutes on llama.cpp. On the same file, its greedy answers to the 5 `--parity` prompts matched llama.cpp's on 4; the fifth parted after 111 characters (abliterated) and 213 (stock).
- **Qwen3.8-Flash-Next Uncensored in its four formats** (the TensorFold, veloGB10 and vLLM v0.33 rows were measured on Nov 28 with each engine started outside the script, with the settings in [Engines](#engines); TensorFold and veloGB10 had their conversation reuse off, vLLM its prefix caching on, and speedtest's prompts start with a random nonce, so no prompt could reuse another's prefix either way): veloGB10 on the EXL3 build decodes fastest, 28-29% faster than PolyStrata, 21-26% faster than TensorFold and about 1.85 times as fast as vLLM v0.33 with fp8 side layers. vLLM v0.33 on the ModelOpt NVFP4 build reads prompts fastest: with the recipe's fp8 side layers 2,796 to 3,332 tok/s, 1.6 to 2.1 times as fast as veloGB10 and 1.6 to 1.8 times as fast as PolyStrata, and the ~208K needle's first token after 73.6 s (108 s on PolyStrata, 162 s on veloGB10, 174 s on TensorFold). The fp8 side layers decode 15-24% faster than the checkpoint as published and read prompts 6-7% faster. TensorFold on the MLX 4-bit build sits between them: 2-6% faster decode than PolyStrata, 6-25% faster prompt reading in the 4k to 32k cases. The formats are different quantizations (EXL3 at 3.05 bits per weight, MLX affine 4-bit, GGUF `IQ3_XXS`, ModelOpt NVFP4 experts with BF16 or fp8 side layers), so these rows compare an engine and its format together, and say nothing about the quality of the answers.
- **Through the script, with conversation reuse on** (Nov 28, 12:30-12:43): in the 4k case, TensorFold read 1,924 tok/s and decoded 70.8, veloGB10 1,689 and 84.5, vLLM v0.33 with fp8 side layers 2,797 and 45.9, within 6% of the rows above. All three reused a conversation's cached prefix on its next turn (TensorFold and veloGB10 to the token; vLLM in whole cache blocks, so it read 1,600 to 2,400 tokens again). A conversation of 16,540 tokens was followed by a turn of about 860 new tokens; that turn's first token came after 0.62 s on TensorFold, 0.59 s on veloGB10 and 0.97 s on vLLM, against 8.50 s, 9.49 s and 5.83 s for a cold prompt of the same length. A third turn, sent after a different conversation had used the server, still resumed: 0.61, 0.65 and 0.71 s.
- **DeepSeek-V4.1-Flash** runs on the Spark at its full 1,048,576-token context: in a run of the abliterated model on Nov 28 (06:13), a ~1M-token needle (922,197 tokens) was found, its first token after 1,787 s (516 tok/s prefill). It decodes at 25-28 tok/s, less than half as fast as Qwen3.8-Flash-Next on PolyStrata, and reads prompts at about a fifth of Flash-Next's speed there. Strata is the only engine here that runs it.
- **The two DeepSeek rows are a pair:** both measured through `--start` and `--speed` on Nov 28 (abliterated at 07:52, stock at 08:12), with the same engine settings and the same download running in the background. In that pair, decode was within 11% either way (the stock model faster in the 4k case, slower at 16k and 32k), prefill was level in the 4k case and 9-23% lower for the stock model at 16k and 32k, and the stock model answered the needle 16% later (450 s against 388 s). Earlier the same day, with the same engine settings but started outside the script, the abliterated model measured 26.7 / 26.5 / 26.1 tok/s decode, 292 / 528 / 606 tok/s prefill and a 309 s needle (06:13), and the stock model 24.5 / 23.9 / 23.5, 222 / 372 / 402 and 444 s (07:29): prefill for the same model varied by up to about 20-25% between runs that day. The two sets differ only in their fourth shard.
- **OrcaRouter's MTP draft model on llama.cpp** (`flashnext-abl`): about 49% faster decoding, 6-7% slower prompt reading (the ~208K needle's first token came 10% later). The script turns it on by default.
- **PolyStrata's refusal-direction projection** (stock Flash-Next) costs 3-5% of decode and 3-4% of prefill speed.
- The stock and abliterated Flash-Next files are different quantizations (ISTA-DASLab's GSQ-RCO `IQ3_S`, OrcaRouter's `IQ3_XXS`), so the speed difference between those two models does not measure abliteration.
- The two Qwen3.8-27B engines read different 4-bit formats of the same OrcaRouter weights (MLX affine 4-bit for TensorFold, NVFP4 for vLLM), so their answers are not expected to match token for token.
- Single runs on one machine; expect a few percent of variation between sessions.

## Features

- Interactive menu, plus flags for scripting
- One registry row per model and engine: each engine downloads the format it reads
- Downloads weights with the Hugging Face CLI (installed into its own venv), pulls the arm64 container images, builds PolyStrata's `strata-poly` for `sm_121`, installs the Strata engine from PolyStrata's signed update feed, installs Strata v0.2.13's signed DeepSeek engine package and TensorFold's and veloGB10's releases after checking each against a SHA-256 pinned in the script, and checks every DeepSeek file against its pinned SHA-256
- Spark-specific settings: `sm_121a` kernels, FlashInfer NVFP4 GEMM, conservative GPU memory use on unified memory
- Speculative decoding: MTP (vLLM and llama.cpp turn it on with a flag; PolyStrata's `strata-poly` uses the GGUF's MTP head on its own; TensorFold and veloGB10 draft with Qwen3.8-Flash-Next's), DFlash2 (TensorFold for Qwen3.8-27B, and PolyStrata's Strata engine for Qwen3.8-Flash-Next), or DSpark (Strata's DeepSeek engine)
- Conversation reuse for agents, which send their whole conversation again every turn: PolyStrata keeps conversations by default, blazux's vLLM recipe turns on vLLM's prefix caching, and the script turns on TensorFold's prompt cache and veloGB10's prefix cache for Qwen3.8-Flash-Next (see [Engines](#engines))
- `--speed` and `--parity` benchmarks of the running model
- Cybench setup and runs, patched to work with a local endpoint, with per-task result tables
- Tiered clean-up: stop servers, remove images, delete weights, or remove everything
- `--dry-run` prints what would happen without changing anything

## Requirements

- An NVIDIA DGX Spark, or another GB10 / aarch64 machine with CUDA 13 drivers (580 or newer). The settings are tuned for the Spark's unified memory.
- Docker with the NVIDIA Container Toolkit (`docker run --gpus all ...` must work), and your user in the `docker` group
- `python3` (with `venv`), `git`, `curl` and `sha256sum`
- For PolyStrata's `strata-poly` (CyberTiel): the CUDA 13 toolkit (`/usr/local/cuda/bin/nvcc`, which DGX OS ships) and `cmake`. The build takes about 10 minutes. Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash need neither: their engines are ready-made releases.
- About 80 GB of free disk per model and engine (Qwen3.8-Flash-Next: 90 GB for the GGUF files, 115 GB for the MLX 4-bit build, 75 GB for the EXL3 build, 175 GB for the ModelOpt NVFP4 build with its converted layers and the image built for it). DeepSeek-V4.1-Flash needs about 262 GB for the first set (257 GB of files: its four shards and the three shared files, plus Strata and its engine) and 21 GB more for the second.
- A Hugging Face account with access granted to gated repos (OrcaRouter's Qwen3.8-27B NVFP4 and Qwen3.8-Flash-Next GGUF). The others are public.

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

The first start can take several minutes while weights load and kernels compile. The script waits until the API answers, then checks that it refuses requests without the key. Picking a model that is not installed yet installs it first.

## Usage

```bash
./spark-llm.sh --list                              # models and the engines each one runs on
./spark-llm.sh --install qwen27b-abl               # the default (fastest) engine's files
./spark-llm.sh --start qwen27b-abl                 # stops any other model first
./spark-llm.sh --start qwen27b-abl --engine vllm   # another engine for the same model
./spark-llm.sh --start qwen27b                     # the stock model, on TensorFold
./spark-llm.sh --install flashnext-abl             # Qwen3.8-Flash-Next Uncensored on PolyStrata (its default)
./spark-llm.sh --start flashnext-abl
./spark-llm.sh --start flashnext-abl --engine llamacpp
./spark-llm.sh --install flashnext-abl --engine velogb10  # the EXL3 build (68 GB) and veloGB10's release
./spark-llm.sh --start flashnext-abl --engine velogb10
./spark-llm.sh --start flashnext-abl --engine tensorfold  # the MLX 4-bit build on TensorFold (after its own --install)
./spark-llm.sh --install flashnext-abl --engine vllm-flash  # the ModelOpt NVFP4 build, blazux's recipe and its image
./spark-llm.sh --start flashnext-abl --engine vllm-flash
VG_PREFIX_CACHE=off ./spark-llm.sh --start flashnext-abl --engine velogb10  # without veloGB10's prefix cache
LLAMA_ABL_MTP=off ./spark-llm.sh --start flashnext-abl --engine llamacpp  # without OrcaRouter's MTP draft model
./spark-llm.sh --start flashnext                   # the stock model, on PolyStrata
PS_REFUSAL_PROJECTION=on ./spark-llm.sh --start flashnext   # the same, with PolyStrata's experimental refusal projection
./spark-llm.sh --install deepseek41-abl            # DeepSeek-V4.1-Flash Uncensored on Strata v0.2.13 (257 GB of files)
./spark-llm.sh --start deepseek41-abl              # with its native 1,048,576-token context
CTX=262144 ./spark-llm.sh --start deepseek41       # the stock model, with a smaller context
./spark-llm.sh --status
./spark-llm.sh --stop

./spark-llm.sh --speed                             # tokens/s of the running model (short, 4k, 16k, 32k cases)
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

The served model names are `Qwen3.8-27B-Uncensored`, `Qwen3.8-27B`, `CyberTiel`, `Qwen3.8-Flash-Next`, `Qwen3.8-Flash-Next-Uncensored`, `DeepSeek-V4.1-Flash-1017` and `DeepSeek-V4.1-Flash-1017-Uncensored`, whichever engine runs them. (PolyStrata's Flash-Next server also answers to its own set name, see below.) Any OpenAI-compatible client works with base URL `http://<spark-ip>:8000/v1` and the key above. On vLLM, Qwen runs with a reasoning parser and tool calling (`qwen3_coder`) turned on. To turn thinking off for a request, send `"chat_template_kwargs": {"enable_thinking": false}`. `--speed` sends it and flags any run that still produced reasoning.

## Engines

### PolyStrata, from a fork

PolyStrata runs mixture-of-experts GGUF models on two engine programs behind one server (`serve/server.py`): `strata-poly` (CyberTiel) and the Strata project's own engine, `strata` (Qwen3.8-Flash-Next). Both are the `polystrata` engine to this script: the same server, API key, PID file, `--stop` and auth check. The model picks the program.

#### CyberTiel on `strata-poly`

PolyStrata runs CyberTiel with its own CUDA engine, `strata-poly`. Upstream ships no aarch64 build of it, and it did not compile on aarch64: two files included the x86-only `<immintrin.h>` for a spin-wait hint. The [`spark-aarch64` branch of our fork](https://github.com/r0yfire/PolyStrata/tree/spark-aarch64) uses PolyStrata's own portable `cpu_relax.hpp` there instead (the same instruction on x86, `yield` on Arm). The script clones that branch at a pinned commit (`PS_REPO`, `PS_REF`), builds `strata-poly` for `sm_121`, exports the model's tokenizer and serves it with PolyStrata's server. `strata-poly` drafts with the GGUF's own MTP head without being asked (its log reports how many drafts were accepted), so the script passes it no speculative-decoding flag, while llama.cpp gets `--spec-type draft-mtp`. With all experts resident on the Spark, PolyStrata's advantage over llama.cpp comes from its kernels, not from expert offloading.

#### Qwen3.8-Flash-Next on the Strata engine

PolyStrata's own `setup.sh` installs Qwen3.8-Flash-Next together with the stock model's 84 GB download, and has no option to install only the engine and the drafter. So the script runs the same steps itself, with PolyStrata's own tools, for both models:

- **The engine:** `strata update --no-control` from the checkout, into `polystrata/terminal/engine-feed` (the step `setup.sh` runs on a Spark). It installs the signed `linux-aarch64` CUDA package from the update feed of the pinned checkout's release; nothing is compiled. The script also installs the cuBLAS and CUDA runtime wheels `setup.sh` installs for this engine, at its pinned versions.
- **The drafter:** `strata drafting fetch`, PolyStrata's tuned DFlash2 drafter for Qwen3.8-Flash-Next (about 0.34 GB, its size and SHA-256 pinned by PolyStrata).
- **The pack:** `tools/iq_pack.py` turns the GGUF's first shard into the pack the engine reads, in `polystrata-data/packs/<model>`. OrcaRouter's file is packed with `--compat-bf16`, as PolyStrata's `docs/ORCA.md` describes: its quantized hyper-connection, attention and PLE projections become the BF16 that PolyStrata's kernels read.
- **The run config** (`polystrata-data/strata-<model>.json`) is written at each start from one template for both models, so they differ only in their files and one engine setting: the engine arguments `setup.sh` writes on a Spark, without the MTP layer's options. OrcaRouter's PLE table is in its first shard, so that shard is both `--native` and `--ple-gguf`; for the stock files it is the second shard. Two settings make PolyStrata's server start DFlash2 with the installed drafter for either model, with its own window and confidence:
  - `model_name` is PolyStrata's name for the set (`qwen3.8-flash-next-iq3_s`, `qwen3.8-flash-next-iq3_xxs`), and the served name is an alias of it, so clients use the served name as with every engine (`/v1/models` lists both). PolyStrata decides which drafter a model gets from that name; for a name it does not know it reads the GGUF header, and its header reader stops at the list of tags at the top of OrcaRouter's file, so it would start that file with no drafter.
  - `user_args` names the drafter (`--draft-model`). Without it, a set with no drafter of its own in PolyStrata's table (`IQ3_XXS` on CUDA) would look for the stock (untuned) DFlash2 pack, which is not downloaded, and start with no drafter.
  - `env` (OrcaRouter's file only) sets `STRATA_HC_Q8=0`. That file stores its hyper-connection projections as Q8_0, so the engine switches on its Q8_0 read of them by itself. On the GB10 that read fails in the engine's first verify window (`operation not supported on global/shared address space`), and the engine exits on the first request. With the setting, the engine reads the pack's BF16 copies instead, as it does for the stock file.

`--dry-run --start flashnext` (or `flashnext-abl`) prints the run config a start would write, once the engine is installed. `PS_REFUSAL_PROJECTION=on` adds PolyStrata's experimental refusal-direction projection to the stock model (`flashnext`): a control vector shipped in the checkout, projected out of layers 4-44 (the flags `setup.sh --experimental-speed-projection on` writes). A request can switch it off with `"experimental_speed_projection": false`. Read `polystrata/terminal/data/experimental-speed-projection/README.md` before using it.

### Strata v0.2.13: DeepSeek-V4.1-Flash

PolyStrata is built on the [Strata](https://github.com/Niko1221/Strata) project, and Strata ran DeepSeek-V4.1-Flash up to release v0.2.13 on an engine of its own, `strata-dsv41`. Later Strata releases, and PolyStrata, no longer run it: PolyStrata refuses a DeepSeek configuration before anything starts (`terminal/strata_control/retired.py`). So the `strata` engine is a checkout of v0.2.13 (`strata-dsv41/strata-0.2.13`, pinned to its commit) with that release's server and tools, and the release's signed DeepSeek engine package for `linux-aarch64` with CUDA 13 (version 0.1.59, built for the GB10). The script runs Strata's own steps for an ISTA-DAS set (`setup.py`'s `install_dsv41`):

- **The engine:** the package's SHA-256 is pinned in the script; then Strata's installer checks the package's signed manifest and every file in it. Every start checks each installed file against that manifest again.
- **The files:** the set's four shards, and three files both sets read, in `models/deepseek41/shared`: ISTA's Engram table (110.6 GB), the DSpark drafter (`dspark-...-Q8_0.gguf`, 8 GB) and the first shard of Unsloth's `UD-Q8_K_XL` set, whose header (5.8 MB) the pack tool reads. Every file is pinned by its SHA-256 (Strata's pins; for AtomicChat's shard 4, the Hub's at the pinned revision), hashed once after download and remembered in `.spark-llm-verified` beside it while its size and modification time stay the same, so later installs and starts do not hash 257 GB again. The Engram is stored on Hugging Face's Xet storage only, and the Hugging Face library refuses plain-HTTP downloads above 50 GB, so the script downloads it with `curl` from the Hub's resolve URL, resuming after any break. A shard the other set's folder already holds (shards 1-3) is linked, not downloaded again.
- **The pack:** `tools/dsv41/pack_gguf.py` with `setup.py`'s arguments for ISTA-DAS sets (`dsv41_pack_cmd`, including the drafter's noise token and Markov rank, which these files do not carry), into `pack/` in the set's folder. The pack is a few MB: the engine reads the GGUF files where they are. The script copies in the expert profile Strata ships for the 2.00 bpw sets (the experts used most, which the engine's expert store starts with) and checks the pack with the engine's `--pack-info`.
- **The run config** (`strata-dsv41/spark-llm-<model>.json`), written at each start: the arguments `setup.py` writes for these sets (`dsv41_config_for`) with this machine's values: the context (`CTX`, 1,048,576 by default), no picture reader (`--vision off`; it is not downloaded), the DSpark drafter with Strata's confidence floor (`--dspark auto --dspark-conf 0.65`), and the larger prompt chunk and engine setting Strata uses on unified-memory GPUs of 64 GB or more (`--prefill-chunk 16384`, `DS4_ASYNC_T=4`). It also states the model's own 6 routed experts per token (`--experts 6`), since the engine's `auto` can pick 5.

The set (138 GB) and the Engram table (110.6 GB) are about 249 GB of files, and the Spark has 128 GB of memory. On the Spark the engine placed about 12,100 of the 15,744 routed experts (108-109 GB) in GPU memory and computes the other 3,600 or so on the CPU, reading them in place from the files. In each run, up to 12% of the experts the first few short requests used came from those, and under 1% from the 4k case on. `--dry-run --start deepseek41-abl` prints the run config a start would write.

### llama.cpp

CyberTiel drafts with the MTP head in its GGUF (`--spec-type draft-mtp`). Both Qwen3.8-Flash-Next models are loaded into memory with `--load-mode none` (this llama.cpp build refuses `--no-mmap`). The stock model (`flashnext`) runs without a drafter; `flashnext-abl` drafts with OrcaRouter's separate MTP draft model (`-md`, with the `--spec-draft-n-max 3` its model card recommends): on the Spark it decoded about 49% faster, for prompt reading 6-7% slower (see Results). `LLAMA_ABL_MTP=off` runs it without the draft model; `--install flashnext-abl --engine llamacpp` downloads the draft model (4.1 GB) either way.

### TensorFold

TensorFold's native release (CUDA kernels embedded in the binary) serves Qwen3.8-27B from an MLX affine 4-bit checkpoint, drafting with DFlash2. The script installs release `TF_VERSION` only if the tarball matches the SHA-256 pinned in the script (`TF_SHA256`; the release's own `.sha256` file is not trusted, since release assets can be replaced), and passes the API key through `--api-key-file`. The native release does not accept `--prefill-fp8` yet (only TensorFold's Python engine does), which is why its prefill trails vLLM on shorter prompts.

It also serves Qwen3.8-Flash-Next (`flashnext-abl --engine tensorfold`) from an MLX affine 4-bit checkpoint with the model's MTP head, which it drafts with. One stream at the full 262,144-token window, with the 8,192-row prompt passes `--parallel 1` gets on one GPU (`TF_FLASHNEXT_CHUNK_SOLO=8192`, set so the run does not depend on that default). On this path TensorFold keeps a conversation only in its prompt cache: its Flash-Next docs say the cache "keeps and resumes a stream's state", and after a reply a background pass reads the reply in as the next turn will send it, so the next turn reads only its new message. With `--prompt-cache-gib 0` an agent's every turn reads its whole conversation again. The cache has a cost on the Spark: TensorFold reads Flash-Next's n-gram tables (29.8 GiB) from the files through the page cache, and its known-issues page measured the first token of a new 65,000-token prompt 12-19% later once 16 GiB of kept states (the default budget) compete with them. So the script gives Flash-Next an 8 GiB budget: room for the state of one full 262,144-token stream (7.99 GiB on the Spark), with less taken from the tables. `TF_PROMPT_CACHE_GIB` sets another budget for any TensorFold model (`0` turns the cache off); Qwen3.8-27B otherwise keeps TensorFold's default.

### veloGB10

[veloGB10](https://github.com/sf-stav/veloGB10) is an inference engine written for the GB10 only. The script installs its release (`v0.8.1`) after checking the tarball against the SHA-256 pinned in the script, and serves Qwen3.8-Flash-Next from Lygodactylus's EXL3 build (`flashnext-abl --engine velogb10`) with the flags of veloGB10's documented one-GB10 command: one lane (`--max-batch 1`), the full 262,144-token window (the most it serves; `CTX` above that is refused), MTP drafting with the model's head (`--mtp auto`), and the n-gram table in RAM when it fits after loading (`--ple-ram auto`; on the Spark it does, 18.5 GiB). No thinking override: the chat template decides, and a request can turn thinking off as with every engine. veloGB10 does not refuse a context it cannot fit: it serves a shorter one. After each start the script compares the context the server reports with `CTX` and warns when it is shorter.

veloGB10 has no API-key option, so it listens on `127.0.0.1:8001` only (`VG_PORT`), and `auth-proxy.py` from this repo (Python standard library only) serves `BIND:PORT`: it requires the bearer key on every route except `GET /health`, compares it in constant time, does not pass it on to the engine, and relays streamed answers piece by piece so time-to-first-token and decode timing are the engine's. The script starts, checks and stops the two together; if either exits during start-up, it stops the other. veloGB10 refuses to start when any `GB10_*` variable is set; the reduced environment native engines get (see Security notes) contains none.

veloGB10's prefix cache is off by default. Its help says it gives "~3x faster follow-up turns on multi-turn/agent traffic", and that it is not bit-exact: a turn read from the cache can word an answer differently from a cold read, while greedy MTP stays lossless. The script turns it on (`VG_PREFIX_CACHE=off` restores the default), with veloGB10's defaults for the rest: up to 4 GiB of intermediate prefill checkpoints (bit-identical to a fresh read), and a prefix pool of up to 32 GiB that keeps a conversation's state when another one takes the lane, without ever taking free memory below 20 GiB.

### vLLM

vLLM drafts with the model's MTP head, 4 tokens by default (`VLLM_SPEC`). The AEON image also contains vLLM's DFlash support, but its DFlash2 drafter failed to start there (`FlashInferImpl.__init__() got an unexpected keyword argument 'use_mm_prefix'`), so it is not the default.

RedHatAI's stock NVFP4 checkpoint (`qwen27b`) records GPTQ activation ordering (`"actorder": "static"`) on its FP8 per-channel group, and the compressed-tensors version in the image refuses that at startup ("Must use group or tensor_group quantization strategy in order to apply activation ordering"). Static ordering only changes how the weights were calibrated, not how they are stored, and vLLM's FP8 and NVFP4 kernels do not read it. So the script writes a copy of `config.json` with only those flags cleared (`$SPARK_LLM_HOME/vllm-config/`) and mounts it over the original in the container. The downloaded files stay as the Hub has them, and models that need no correction get none. `--speed` records the correction as `config_fix`.

### vLLM v0.33 for Qwen3.8-Flash-Next (`vllm-flash`)

Qwen3.8-Flash-Next's ModelOpt NVFP4 checkpoint is 126 GiB, more than the Spark's memory, and the AEON vLLM image would hold all of it, including its 47.7 GiB n-gram table. [blazux's qwen3.8-Flash-DGX recipe](https://github.com/blazux/qwen3.8-Flash-DGX) builds vLLM v0.33 with patches that map that table from disk instead, and documents this checkpoint's layout (RadixArk's, of which jpezzulli's abliterated build is a copy) on one Spark. It is a separate engine to this script, `vllm-flash`, so the `vllm` engine and its image stay as they are. `--install flashnext-abl --engine vllm-flash`:

- clones the recipe at a pinned commit, checks that its Dockerfile starts from the `vllm/vllm-openai:v0.33.0` image pinned by digest (the Dockerfile also pins every file it adds by checksum), and builds the image from it (`qwen38-flash-dgx:v0.33-01731a10`, 26 GB);
- lays the checkpoint out as the recipe's `serve.sh` reads it, a Hugging Face cache under `$SPARK_LLM_HOME/blazux-hf` whose small files are copies and whose weight files are links to the model folder, mounted read-only in the container (nothing large is copied);
- converts the checkpoint's BF16 side layers (attention, Gated DeltaNet and shared-expert projections) to blockwise fp8 with the recipe's own converter, once (13 GB, written in under a minute on the Spark). The routed experts and the MTP head are untouched.

`--start` runs the recipe's `serve.sh`, unmodified, with the fp8 side layers (`MODE=hybrid`) and the recipe's "fast" profile at the native 262,144-token window (the most it serves here; `CTX` above it is refused): up to 16 sequences, a bfloat16 recurrent-state cache, a 13 GiB fp8 KV cache, 3 MTP drafts with the draft head's output layer in NVFP4, 16,384-token batches. `serve.sh` turns on vLLM's prefix caching, so an agent's next turn reads only its new tokens. The script adds its own label to the container (so `--stop` and `--status` find it), the key middleware every vLLM server here gets (see Security notes), and the address from `BIND`; it runs `serve.sh` with a reduced environment, because `serve.sh` reads many optional variables from it (a `CC` set for a C compiler would replace vLLM's compile settings). `serve.sh` starts the container with `--restart unless-stopped`: `--stop` removes it, but a container still running when the machine shuts down comes back with Docker. The container's log: `docker logs llm-flashnext-abl`.

## Configuration

All settings are environment variables with sensible defaults:

| Variable | Default | Purpose |
|---|---|---|
| `SPARK_LLM_HOME` | `~/spark-llm` | Where weights, engines, venv, Cybench, results and the API key live |
| `PORT` | `8000` | Host port for the API |
| `BIND` | `0.0.0.0` | Address the API listens on: `0.0.0.0` for every interface (LAN clients), `127.0.0.1` for this machine only. IPv4 only. See [Security notes](#security-notes). |
| `CTX` | `262144` (DeepSeek-V4.1-Flash: `1048576`) | Context window. When set, every model gets it; DeepSeek-V4.1-Flash takes at most 1,048,576. |
| `VLLM_GPU_UTIL` | `0.70` | vLLM memory share. Keep it at or below 0.8 on unified memory. |
| `VLLM_SPEC` | `{"method":"mtp","num_speculative_tokens":4}` | vLLM's `--speculative-config`. `VLLM_SPEC=` (empty) turns speculative decoding off. |
| `VLLM_IMAGE` | `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` | vLLM image built for the Spark |
| `LLAMA_IMAGE` | `ghcr.io/ggml-org/llama.cpp:server-cuda13` | llama.cpp server image |
| `PS_REPO` / `PS_REF` | our fork / a pinned commit | PolyStrata source to build (and, for Qwen3.8-Flash-Next, the release whose engine and drafter it installs) |
| `PS_REFUSAL_PROJECTION` | `off` | `on`: PolyStrata's experimental refusal-direction projection on the stock Qwen3.8-Flash-Next (`flashnext` on `polystrata` only) |
| `LLAMA_ABL_MTP` | `on` | llama.cpp drafts with OrcaRouter's MTP draft model (`flashnext-abl` on `llamacpp` only); `off`: no drafter |
| `TF_VERSION` / `TF_SHA256` | `1.3.6` / that release's tarball hash | TensorFold release, and the SHA-256 its tarball must match. Another version needs its own `TF_SHA256`. |
| `TF_PROMPT_CACHE_GIB` | – (Qwen3.8-Flash-Next: `8`) | TensorFold's prompt cache in GiB, which lets a conversation's next turn read only its new tokens; `0` turns it off. Unset: TensorFold's own default, except 8 GiB for Qwen3.8-Flash-Next (see [TensorFold](#tensorfold)). |
| `VG_PREFIX_CACHE` | `on` | veloGB10's prefix cache (reuse of a conversation's prompt on its next turn); `off` is veloGB10's own default |
| `VG_PORT` | `8001` | veloGB10's own port, on `127.0.0.1` only; `auth-proxy.py` serves `PORT` in front of it |
| `VF_KV_DTYPE` | `fp8_e4m3` | `vllm-flash`'s KV cache: `fp8_e4m3` as measured (the recipe's "fast" profile), or `auto` for bf16 (not measured here) |
| `DRAFTER_27B_REPO` | `z-lab/Qwen3.8-27B-DFlash2` | DFlash2 drafter for Qwen3.8-27B |
| `VLLM_EXTRA` / `LLAMA_EXTRA` / `TF_EXTRA` / `VG_EXTRA` | – | Extra arguments for vLLM, llama.cpp, TensorFold or veloGB10 (`QWEN_EXTRA` / `TIEL_EXTRA` still work) |
| `SPEED_RUNS` / `SPEED_NEEDLE` | `3` / `0` | Runs per prompt size; needle prompt size in tokens (0 skips it) |
| `HF_TOKEN` | – | Hugging Face token. Otherwise the `hf auth login` cache is used. |
| `CYBENCH_ITER` | `15` | Agent iterations per task |
| `CYBENCH_TASK_TIMEOUT` | `21600` | Per-task wall-clock safety net, in seconds |
| `CYBENCH_ONLY` | – | Comma-separated task numbers to run, e.g. `6,7,9` |
| `CYBENCH_MAX_IN` / `CYBENCH_MAX_OUT` | `120000` / `32768` | Token limits per iteration |

## Speed test

`speedtest.py` (standard-library Python, no dependencies) drives any OpenAI-compatible server:

- Each prompt is real source code (the Python standard library on the machine, read in a fixed order) and starts with a random nonce, so no prefix or conversation cache can skip the prefill. The cases (`short`, `4k`, `16k`, `32k`, ...) are named by nominal size; each run records the prompt tokens the engine actually counted.
- Token counts come only from the server's own `usage` (`stream_options.include_usage`). A run without `usage` gets no rates (shown as `n/a`, flagged `no_usage`) rather than a count of streamed chunks, which would understate decode for engines that send several drafted tokens per chunk.
- Runs are flagged, and left out of the medians, when the model produced reasoning even though thinking was turned off: streamed reasoning, a `<think>` tag in the answer, `reasoning_tokens` in `usage`, or more completion tokens than the streamed text can hold, or under 2 characters per token (a heuristic): signs of hidden reasoning, which would make prefill look slow and decode look fast. Flagged runs are printed with a warning and kept in the JSON.
- `--needle N` checks long context; `--parity` saves greedy answers to fixed prompts; `--table FILES...` prints a markdown table of result files, one row per label and engine configuration.

`--speed` writes results to `$SPARK_LLM_HOME/speed/<model>-<engine>-<timestamp>.json`, including the engine configuration the model was started with (`meta`: image and image ID, `VLLM_SPEC`, `CTX`, PolyStrata commit, the Strata engine's version, drafter and engine settings, `PS_REFUSAL_PROJECTION`, `LLAMA_ABL_MTP`, Strata v0.2.13's commit, DeepSeek engine version and arguments, TensorFold version, extra arguments).

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
├── .active             # which model and engine are running, and the address it listens on
├── .active.json        # the running engine's configuration (recorded in --speed results)
├── models/             # downloaded weights, one folder per model and format (DeepSeek: deepseek41/<set>/ with its pack, and shared/)
├── polystrata/         # PolyStrata checkout, its strata-poly build and the Strata engine (terminal/engine-feed)
├── polystrata-data/    # packs, run configs and the DFlash2 drafter (dflash2/)
├── strata-dsv41/       # Strata v0.2.13 checkout, its DeepSeek engine package (engine-dsv41/) and run configs
├── tensorfold/         # TensorFold release
├── velogb10/           # veloGB10 release
├── blazux/             # blazux's qwen3.8-Flash-DGX recipe (vllm-flash), at its pinned commit
├── blazux-hf/          # the Hugging Face cache layout its serve.sh reads, with the fp8 side layers
├── vllm-auth/          # the key-checking middleware mounted into the vLLM container
├── vllm-config/        # corrected config.json copies mounted over a checkpoint's own, where one is needed
├── venv/               # Hugging Face CLI
├── speed/              # speed and parity results (JSON)
├── logs/               # native engines' server logs (veloGB10: velogb10-<model>.log and velogb10-proxy.log)
├── cybench/            # patched Cybench checkout (.env holds the key, mode 600)
└── results/            # per-run Cybench logs and tasks.tsv
```

## Troubleshooting

- **`Access denied ... requires approval` when downloading.** Request access on the model page, then set `HF_TOKEN` or run `hf auth login`.
- **`cannot talk to Docker`.** Add your user to the `docker` group and log in again.
- **`port 8000 is still in use after stopping everything`.** Something outside the script holds the port: `ss -ltnp '( sport = :8000 )'` shows what.
- **`did not refuse requests without a valid key`.** A server answered an unauthenticated request after start, so the script stopped it. The lines above the error name each path and status. Do not work around it: the engine's key handling has to be fixed first.
- **PolyStrata, Strata, TensorFold or veloGB10 exits at start.** Read `~/spark-llm/logs/polystrata-server.log` (the engine's own log: `~/spark-llm/logs/polystrata-<model>.log`), `~/spark-llm/logs/strata-server.log` (DeepSeek's engine: `~/spark-llm/logs/strata-<model>.log`), `~/spark-llm/logs/tensorfold-<model>.log`, or `~/spark-llm/logs/velogb10-<model>.log` and `velogb10-proxy.log` (its key proxy). A failed PolyStrata build logs to `~/spark-llm/logs/polystrata-build.log`. For Qwen3.8-Flash-Next, the server log's `[strata] drafting:` line says which drafter the engine started with.
- **`PolyStrata is serving a model: run --stop first` during an install.** PolyStrata's updater does not install the Strata engine while one of its servers runs. Stop the model, install, start again. The native engines start with a reduced environment (see Security notes); a variable they need that is not on the list in `scrub_env` has to be added there.
- **`vllm-flash` exits at start.** Read `docker logs llm-flashnext-abl` (the container is removed after a failed start; the last 60 lines are printed first), `~/spark-llm/logs/vllm-flash-build.log` for the image build and `~/spark-llm/logs/vllm-flash-fp8convert.log` for the conversion.
- **Out-of-memory errors or a slow system.** Lower `CTX` (or `VLLM_GPU_UTIL` for vLLM). Unified memory is shared with the OS.
- **A Cybench task shows `NOT RUN`.** Read `results/<run>/task-N.log`. The usual causes are an x86-only challenge, a failed Docker build, or a leftover container holding a port. The script removes a task's containers after each task.

## Security notes

- By default the API listens on every interface (`BIND=0.0.0.0`), so other machines on your network can reach it. `BIND=127.0.0.1` keeps it to this machine; Cybench then cannot reach it, because its agent container connects over the LAN address.
- Firewalls: ports that Docker publishes (both vLLM engines and llama.cpp) bypass ufw. Docker redirects their traffic in the `nat` table before it reaches the `INPUT` chain ufw uses, so `ufw deny 8000` does not block them. Put such rules in Docker's `DOCKER-USER` chain instead, for example `sudo iptables -I DOCKER-USER -p tcp -m conntrack --ctorigdstport 8000 ! -s 192.168.1.0/24 -j DROP` to allow only your LAN (see Docker's "Packet filtering and firewalls" docs). The native engines (PolyStrata, Strata, TensorFold, and veloGB10's key proxy) listen on the host directly, so ufw does cover them. veloGB10 itself listens on `127.0.0.1` only.
- Every inference route of every engine needs the key. vLLM's own key check covers only `/v1`, `/v2`, `/inference` and `/cohere`, and its `POST /invocations` would serve the model without one, so the script mounts a small middleware into the vLLM container of either vLLM engine (`$SPARK_LLM_HOME/vllm-auth/spark_llm_auth.py`, loaded with `--middleware`) that requires the key on every route except `/health` and `/ping`. After every start the script proves it: POSTs without a key and with a wrong key to `/v1/chat/completions` (and, on vLLM, `/invocations`) must get 401 or 403, and other inference paths must refuse too or not exist. Otherwise the script stops the server and exits with an error. Health checks, and llama.cpp's model list (`/v1/models`), answer without a key.
- The API key, and Cybench's `.env` that contains it, are written with mode `600`, are never printed (not even by `--dry-run`), and never appear on a command line: vLLM reads it from `VLLM_API_KEY` in its container's environment, llama.cpp and TensorFold from the key file (`--api-key-file`; read-only mount for llama.cpp), `auth-proxy.py` (in front of veloGB10, which never sees the key) from the key file too, PolyStrata and Strata from `STRATA_API_KEY`, and the script's own `curl` checks from standard input (`curl -K -`). Members of the `docker` group can still read it with `docker inspect`, but that group is equivalent to root anyway.
- The script never stores or logs your Hugging Face token. The native engines start with a reduced environment (paths, locale, CUDA/NVIDIA, Python and engine variables), so your token and other credentials exported in the shell do not reach them. PolyStrata's and Strata's install tools (updater and installer, drafter fetch, pack tools) run like `pip` does, with your shell's environment.
- What is pinned: PolyStrata's source (a commit of our fork, `PS_REF`); the Strata engine (the signed package of that release's update feed: PolyStrata's updater checks the signature, and every start checks each file against the package's manifest); the DFlash2 drafter (PolyStrata's pinned size and SHA-256; a start refuses a file that does not match); the stock Qwen3.8-Flash-Next files (the Hugging Face revision PolyStrata pins); the stock Qwen3.8-27B files (the revisions measured); Strata v0.2.13 (its commit), its DeepSeek engine package (the SHA-256 recorded in the script, then the package's signed manifest; every start checks each file against the manifest); every DeepSeek file (its SHA-256: Strata's pins, and the Hub's at the pinned revision for AtomicChat's shard 4); TensorFold's release tarball (its SHA-256, `TF_SHA256`, recorded in the script; another `TF_VERSION` needs its own `TF_SHA256`); veloGB10's release tarball (its SHA-256, recorded in the script); blazux's recipe for `vllm-flash` (its commit, and through it the base image's digest and the checksum of every file its Dockerfile adds; the image is built here); and the MLX 4-bit, EXL3 and ModelOpt NVFP4 builds of Qwen3.8-Flash-Next Uncensored (the revisions measured). What is not: the container images are pulled by tag (set `VLLM_IMAGE` / `LLAMA_IMAGE` to `image@sha256:<digest>` to pin them; `docker image inspect --format '{{index .RepoDigests 0}}' <image>` prints the digest you have), the Hugging Face CLI and `ninja` come from PyPI unpinned, PolyStrata's and Strata's Python requirements and the CUDA wheels by version but without hashes, OrcaRouter's Qwen3.8-Flash-Next files at the repo's current revision, and vLLM runs with `--trust-remote-code` on the model repo.
- Cybench task containers run with `--privileged`, as upstream does. Run benchmarks only on a machine you're comfortable using for that.

## Acknowledgements

- Model authors: [orcarouter](https://huggingface.co/orcarouter) (Qwen3.8-27B Uncensored, Qwen3.8-Flash-Next Uncensored) and [peculiar-ragdoll](https://huggingface.co/peculiar-ragdoll) (CyberTiel), building on the Qwen base models; [AtomicChat](https://huggingface.co/AtomicChat) (DeepSeek-V4.1-Flash-1017 Uncensored), building on DeepSeek's model; [ISTA-DASLab](https://huggingface.co/ISTA-DASLab) (Qwen3.8-Flash-Next and DeepSeek-V4.1-Flash-1017 GSQ-RCO GGUF, the Engram table); [Unsloth](https://huggingface.co/unsloth) (the DeepSeek DSpark drafter GGUF and metadata shard); [RedHatAI](https://huggingface.co/RedHatAI) and [mlx-community](https://huggingface.co/mlx-community) (stock Qwen3.8-27B NVFP4 and MLX 4-bit); [z-lab](https://huggingface.co/z-lab) (Qwen3.8-27B DFlash2 drafter) and [incoai](https://huggingface.co/incoai) (Qwen3.8-Flash-Next DFlash2 drafter, tuned by PolyStrata); [yikoudazhong](https://huggingface.co/yikoudazhong) (Qwen3.8-Flash-Next Uncensored, MLX 4-bit), [Lygodactylus](https://huggingface.co/Lygodactylus) (Qwen3.8-Flash-Next Uncensored, EXL3) and [jpezzulli](https://huggingface.co/jpezzulli) (Qwen3.8-Flash-Next Uncensored, ModelOpt NVFP4)
- [vLLM](https://github.com/vllm-project/vllm), the AEON DGX Spark build, and [blazux's qwen3.8-Flash-DGX recipe](https://github.com/blazux/qwen3.8-Flash-DGX)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [PolyStrata](https://github.com/simplexai-labs/PolyStrata) and [Strata](https://github.com/Niko1221/Strata)
- [TensorFold](https://github.com/ashhart/TensorFold)
- [veloGB10](https://github.com/sf-stav/veloGB10)
- [Cybench](https://github.com/andyzorigin/cybench)

## License

[MIT](LICENSE). The models, engines, container images and Cybench each have their own licenses.
