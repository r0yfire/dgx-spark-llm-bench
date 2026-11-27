#!/usr/bin/env python3
"""Engine-neutral tokens-per-second benchmark for an OpenAI-compatible server.

Every engine (vLLM, llama.cpp, PolyStrata) is measured with the same prompts, the
same streaming client and the same arithmetic, so the numbers are comparable:

  prefill tok/s = prompt_tokens / time-to-first-token
  decode  tok/s = (completion_tokens - 1) / (last token time - first token time)

Token counts come from the server's own `usage` (stream_options.include_usage).
Each prompt starts with a random nonce so no prefix / conversation cache can skip
the prefill. Prompt text is real source code (this machine's Python stdlib, read
in sorted order), so drafting (MTP) sees realistic text, not repeated filler.

Usage:
  speedtest.py --base-url http://127.0.0.1:8000/v1 --key-file ~/spark-llm/.api_key \
      --label qwen27b-vllm --cases short,4k,16k,32k --runs 3 --out results.json
  speedtest.py ... --needle 200000     # long-context check: hide a fact, ask for it back
"""
import argparse, glob, http.client, json, os, statistics, sys, time, uuid
from urllib.parse import urlparse

CASES = {"short": 0, "4k": 4096, "16k": 16384, "32k": 32768, "64k": 65536, "128k": 131072}
# Fixed prompts for --parity: no nonce, temperature 0, so two engines serving the same weights should agree.
PARITY_PROMPTS = [
    "Write a Python function that returns the n-th Fibonacci number iteratively.",
    "Explain what a SQL injection is in two sentences.",
    "List the first five prime numbers and say why 1 is not prime.",
    "Translate to French: The quick brown fox jumps over the lazy dog.",
    "What does the Linux command `chmod 640 file` do?",
]
CHARS_PER_TOKEN = 3.3   # rough, for sizing; the real count is reported by the server


def corpus(min_chars):
    """Deterministic real text: the stdlib's .py files, sorted, concatenated."""
    paths = sorted(glob.glob(os.path.join(os.path.dirname(os.__file__), "*.py")))
    out, n = [], 0
    for p in paths * 50:
        try:
            s = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        out.append(f"\n# ---- file: {os.path.basename(p)} ----\n{s}")
        n += len(out[-1])
        if n >= min_chars:
            break
    return "".join(out)[:min_chars]


def stream_chat(base, key, model, messages, max_tokens, extra, timeout):
    """One streamed chat request. Returns timings + server usage."""
    u = urlparse(base)
    conn_cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    conn = conn_cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout)
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0,
            "stream": True, "stream_options": {"include_usage": True}}
    body.update(extra)
    hdr = {"Content-Type": "application/json", "Authorization": f"Bearer {key}"}
    t0 = time.perf_counter()
    conn.request("POST", u.path.rstrip("/") + "/chat/completions", json.dumps(body), hdr)
    r = conn.getresponse()
    if r.status != 200:
        raise RuntimeError(f"HTTP {r.status}: {r.read()[:500]!r}")
    t_first = t_last = None
    usage, text, reasoning, chunks = None, [], [], 0
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        ev = json.loads(data)
        if ev.get("usage"):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = ch.get("delta") or {}
            piece = d.get("content") or ""
            rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
            if piece or rpiece:
                now = time.perf_counter()
                t_first = t_first or now
                t_last = now
                chunks += 1
                text.append(piece)
                reasoning.append(rpiece)
    conn.close()
    t_end = time.perf_counter()
    if t_first is None:
        raise RuntimeError("no tokens streamed")
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens") or chunks
    return {"ttft_s": t_first - t0, "total_s": t_end - t0, "prompt_tokens": pt, "completion_tokens": ct,
            "usage_reported": usage is not None, "chunks": chunks,
            "prefill_tps": (pt / (t_first - t0)) if pt else None,
            "decode_tps": ((ct - 1) / (t_last - t_first)) if ct > 1 and t_last > t_first else None,
            "text": "".join(text), "reasoning_chars": sum(map(len, reasoning))}


def messages_for(case_tokens, nonce):
    """A code-explanation task over `case_tokens` of real source (none for 'short')."""
    head = f"Request {nonce}.\n"
    if not case_tokens:
        return [{"role": "user", "content": head + "Write a detailed, well-structured explanation of how "
                 "TCP congestion control works (slow start, congestion avoidance, fast retransmit, fast "
                 "recovery, CUBIC and BBR), with a short worked example."}]
    src = corpus(int(case_tokens * CHARS_PER_TOKEN))
    return [{"role": "user", "content": head + "Here is some Python source code:\n\n" + src +
             "\n\nExplain in detail what the code above does, file by file."}]


def needle_messages(n_tokens, nonce):
    """Long-context check: a fact hidden at one third of the prompt, asked for at the end."""
    code = uuid.uuid4().hex[:10].upper()
    hay = corpus(int(n_tokens * CHARS_PER_TOKEN))
    cut = len(hay) // 3
    hay = hay[:cut] + f"\n# NOTE: the vault access code is {code}. Remember it.\n" + hay[cut:]
    return code, [{"role": "user", "content": f"Request {nonce}.\n" + hay +
                   "\n\nWhat is the vault access code mentioned in a NOTE comment above? Answer with the code only."}]


def table(paths):
    """Markdown table of median prefill/decode per case (and the needle result) from result JSON files."""
    rows, cases = {}, []
    for p in paths:
        d = json.load(open(p))
        r = rows.setdefault(d["label"], {})
        for c, v in d.get("cases", {}).items():
            cases += [c] if c not in cases else []
            r[c] = (v.get("median_prefill_tps"), v.get("median_decode_tps"))
        if "needle" in d:
            n = d["needle"]
            r["needle"] = f"{n['prompt_tokens']:,} tok, {'found' if n.get('found') else 'MISSED'}, TTFT {n['ttft_s']:.0f}s"
    fmt = lambda x, nd: "-" if x is None else f"{x:,.{nd}f}"
    print("| label | " + " | ".join(f"{c} decode / prefill" for c in cases) + " | ~200K needle |")
    print("|---|" + "---:|" * len(cases) + "---|")
    for label, r in rows.items():
        cells = [f"{fmt(r[c][1], 1)} / {fmt(r[c][0], 0)}" if c in r else "-" for c in cases]
        print(f"| {label} | " + " | ".join(cells) + f" | {r.get('needle', '-')} |")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--table":
        return table(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key-file", default=os.path.expanduser("~/spark-llm/.api_key"))
    ap.add_argument("--model", help="model name (default: first from /v1/models)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--cases", default="short,4k,16k,32k")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--extra-body", default="{}", help="JSON merged into each request (thinking switches etc.)")
    ap.add_argument("--needle", type=int, default=0, help="run one long-context needle check of N tokens instead")
    ap.add_argument("--parity", action="store_true",
                    help="greedy answers to fixed prompts (64 tokens each), saved to --out: compare two engines on one model")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--out")
    a = ap.parse_args()
    key = open(os.path.expanduser(a.key_file)).read().strip() if os.path.exists(os.path.expanduser(a.key_file)) else "none"
    extra = json.loads(a.extra_body)
    model = a.model
    if not model:
        u = urlparse(a.base_url)
        c = http.client.HTTPConnection(u.hostname, u.port, timeout=30)
        c.request("GET", u.path.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
        model = json.loads(c.getresponse().read())["data"][0]["id"]
    result = {"label": a.label, "model": model, "base_url": a.base_url, "extra_body": extra,
              "max_tokens": a.max_tokens, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cases": {}}

    if a.parity:
        result["parity"] = []
        for i, p in enumerate(PARITY_PROMPTS):
            r = stream_chat(a.base_url, key, model, [{"role": "user", "content": p}], 64, extra, a.timeout)
            result["parity"].append({"prompt": p, "text": r["text"], "completion_tokens": r["completion_tokens"]})
            print(f"  [{i}] {r['text'][:100]!r}")
    elif a.needle:
        code, msgs = needle_messages(a.needle, uuid.uuid4().hex)
        r = stream_chat(a.base_url, key, model, msgs, 64, extra, a.timeout)
        r["found"] = code in r["text"]
        r["expected"] = code
        result["needle"] = dict(r)
        print(f"{a.label} needle {a.needle}: prompt {r['prompt_tokens']} tok, TTFT {r['ttft_s']:.1f}s, "
              f"prefill {r['prefill_tps'] or 0:.0f} tok/s, found={r['found']} answer={r['text'][:60]!r}")
    else:
        stream_chat(a.base_url, key, model, messages_for(0, "warmup"), 32, extra, a.timeout)   # warm-up, not recorded
        for case in a.cases.split(","):
            runs = []
            for i in range(a.runs):
                r = stream_chat(a.base_url, key, model, messages_for(CASES[case], uuid.uuid4().hex),
                                a.max_tokens, extra, a.timeout)
                r.pop("text")
                runs.append(r)
                print(f"  {a.label} {case} run {i+1}: prompt {r['prompt_tokens']} tok, TTFT {r['ttft_s']:.2f}s, "
                      f"prefill {r['prefill_tps'] or 0:.0f} tok/s, decode {r['decode_tps'] or 0:.1f} tok/s "
                      f"({r['completion_tokens']} tok, usage={'yes' if r['usage_reported'] else 'NO'})", flush=True)
            med = lambda k: statistics.median([x[k] for x in runs if x[k]]) if any(x[k] for x in runs) else None
            result["cases"][case] = {"runs": runs, "median_prefill_tps": med("prefill_tps"),
                                     "median_decode_tps": med("decode_tps"), "median_ttft_s": med("ttft_s")}
        print(f"\n{a.label} ({model}) medians:")
        for case, c in result["cases"].items():
            print(f"  {case:>5}: prefill {c['median_prefill_tps'] or 0:8.0f} tok/s   decode {c['median_decode_tps'] or 0:6.1f} tok/s")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
