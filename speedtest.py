#!/usr/bin/env python3
"""Engine-neutral tokens-per-second benchmark for an OpenAI-compatible server.

Every engine (vLLM, llama.cpp, PolyStrata, TensorFold) is measured with the same
prompts, the same streaming client and the same arithmetic, so the numbers are comparable:

  prefill tok/s = prompt_tokens / time-to-first-token
  decode  tok/s = (completion_tokens - 1) / (last token time - first token time)

Token counts come only from the server's own `usage` (stream_options.include_usage).
A run without usage gets no rates (None, flag "no_usage"): counting SSE chunks instead
would understate decode for engines that stream several drafted tokens per chunk.

The caller turns thinking off (chat_template_kwargs.enable_thinking=false). A run that
still produced reasoning is flagged and left out of the medians, because hidden
reasoning inflates the time to first token (prefill looks slow) and adds tokens to
completion_tokens that the streamed window never saw (decode looks fast). Flags:
  no_usage            the server sent no usage
  reasoning_streamed  reasoning deltas (reasoning_content / reasoning) arrived
  think_tag           <think> or </think> appeared in the answer
  reasoning_tokens    usage.completion_tokens_details.reasoning_tokens > 0
  hidden_tokens       completion_tokens > streamed characters + 2: more tokens than the
                      visible text can hold, so some were generated but not streamed
  low_chars_per_token under 2 streamed characters per completion token (64+ tokens): a
                      heuristic for shorter hidden reasoning; prose and code answers run
                      at roughly 3-4.5 characters per token

Prompt cases are named by their nominal size (CHARS_PER_TOKEN characters per token);
the real prompt_tokens each engine's tokenizer counts is in every run's record.
Each prompt starts with a random nonce so no prefix / conversation cache can skip
the prefill. Prompt text is real source code (this machine's Python stdlib, read
in sorted order), so drafting (MTP) sees realistic text, not repeated filler.

Usage:
  speedtest.py --base-url http://127.0.0.1:8000/v1 --key-file ~/spark-llm/.api_key \
      --label qwen27b-vllm --cases short,4k,16k,32k --runs 3 --out results.json \
      [--meta '{"vllm_spec": "...", "ctx": "262144"}']   # engine config, recorded in the result
  speedtest.py ... --needle 200000     # long-context check: hide a fact, ask for it back
  speedtest.py --table results/*.json  # markdown table of result files
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
CHARS_PER_TOKEN = 3.3   # sizes the nominal cases; real tokenizers count fewer tokens (see each run's prompt_tokens)


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


def connect(url, timeout):
    """HTTP(S) connection for a base URL (port defaults by scheme)."""
    u = urlparse(url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    return cls(u.hostname, u.port or (443 if u.scheme == "https" else 80), timeout=timeout), u


def _json_or_none(s):
    try:
        return json.loads(s)
    except ValueError:
        return None


class SSE:
    """OpenAI-style server-sent events: yields (arrival time, payload) per `data:` payload.

    Keep-alives (empty data), comments and other fields are skipped, a payload split over
    several `data:` lines is joined, `[DONE]` ends the stream. Payloads that never parse
    as JSON are counted in `bad` instead of aborting the run."""

    def __init__(self, lines):
        self.lines, self.bad = lines, 0

    def __iter__(self):
        pending = []
        for raw in self.lines:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if not line.strip():                    # blank line: end of an event
                self.bad += bool(pending)
                pending = []
                continue
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if not data:
                continue
            ev = _json_or_none(data)
            if ev is None and pending:              # continuation of a multi-line payload?
                ev = _json_or_none("\n".join(pending + [data]))
                if ev is None:
                    pending.append(data)
                    continue
            elif ev is None:
                pending.append(data)
                continue
            elif pending:                           # a complete payload after an unparsable one
                self.bad += 1
            pending = []
            yield time.perf_counter(), ev
        self.bad += bool(pending)


def run_flags(usage, text, reasoning_chars, completion_tokens):
    """Why a run cannot be compared with the others (empty: it can). See the module docstring."""
    flags = []
    if usage is None:
        flags.append("no_usage")
    if reasoning_chars:
        flags.append("reasoning_streamed")
    if "<think>" in text or "</think>" in text:
        flags.append("think_tag")
    details = (usage or {}).get("completion_tokens_details") or {}
    if isinstance(details, dict) and details.get("reasoning_tokens"):
        flags.append("reasoning_tokens")
    if completion_tokens is not None and completion_tokens > len(text) + 2:
        flags.append("hidden_tokens")
    elif completion_tokens is not None and completion_tokens >= 64 and len(text) / completion_tokens < 2.0:
        flags.append("low_chars_per_token")
    return flags


def stream_chat(base, key, model, messages, max_tokens, extra, timeout):
    """One streamed chat request. Returns timings, the server's usage and the run's flags."""
    conn, u = connect(base, timeout)
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
    sse = SSE(r)
    for now, ev in sse:
        if not isinstance(ev, dict):
            sse.bad += 1
            continue
        if ev.get("error"):
            raise RuntimeError(f"server error in the stream: {json.dumps(ev['error'])[:500]}")
        if isinstance(ev.get("usage"), dict):
            usage = ev["usage"]
        for ch in ev.get("choices") or []:
            d = (ch.get("delta") if isinstance(ch, dict) else None) or {}
            piece = d.get("content") or ""
            rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
            if piece or rpiece:
                t_first = t_first or now
                t_last = now
                chunks += 1
                text.append(piece)
                reasoning.append(rpiece)
    conn.close()
    t_end = time.perf_counter()
    if t_first is None:
        raise RuntimeError("no tokens streamed")
    text = "".join(text)
    rchars = sum(map(len, reasoning))
    pt = (usage or {}).get("prompt_tokens")
    ct = (usage or {}).get("completion_tokens")          # never the chunk count: see the module docstring
    flags = run_flags(usage, text, rchars, ct)
    return {"ttft_s": t_first - t0, "total_s": t_end - t0, "prompt_tokens": pt, "completion_tokens": ct,
            "usage_reported": usage is not None, "chunks": chunks, "bad_events": sse.bad,
            "prefill_tps": (pt / (t_first - t0)) if pt else None,
            "decode_tps": ((ct - 1) / (t_last - t_first)) if ct and ct > 1 and t_last > t_first else None,
            "chars_per_token": (len(text) / ct) if ct else None,
            "text": text, "reasoning_chars": rchars, "flags": flags}


def fmt_rate(x, nd):
    return "n/a" if x is None else f"{x:,.{nd}f}"


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


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


def _short(v):
    v = v if isinstance(v, str) else json.dumps(v, sort_keys=True)
    return v if len(v) <= 40 else v[:37] + "..."


def table(paths):
    """Markdown table of median prefill/decode per case (and the needle result) from result JSON files.

    Rows are keyed by label AND engine config (`meta`): files with one label but different
    configs (MTP 2 vs 4 drafts, another CTX, ...) get separate rows, named by the meta keys
    that differ. Files with the same label and config (a speed file and its needle file)
    share a row."""
    docs = []
    for p in paths:
        with open(p) as f:
            docs.append(json.load(f))
    by_label = {}
    for d in docs:
        by_label.setdefault(d.get("label", "?"), []).append(d)
    rows, cases = {}, []
    for label, ds in by_label.items():
        metas = [d.get("meta") or {} for d in ds]
        keys = sorted({k for m in metas for k in m} - {"started"})
        varying = [k for k in keys if len({json.dumps(m.get(k), sort_keys=True) for m in metas}) > 1]
        for d, m in zip(ds, metas):
            name = label + (" [" + ", ".join(f"{k}={_short(m.get(k))}" for k in varying) + "]" if varying else "")
            r = rows.setdefault(name, {})
            for c, v in d.get("cases", {}).items():
                cases += [c] if c not in cases else []
                runs = v.get("runs") or []
                used = v.get("runs_used", len(runs))
                note = f" ({used}/{len(runs)} runs; {','.join(v.get('flags') or [])})" if used < len(runs) else ""
                r[c] = f"{fmt_rate(v.get('median_decode_tps'), 1)} / {fmt_rate(v.get('median_prefill_tps'), 0)}{note}"
            if "needle" in d:
                n = d["needle"]
                pt = n.get("prompt_tokens")
                ttft = n.get("ttft_s")
                flags = n.get("flags") or []
                r["needle"] = (f"{'?' if pt is None else f'{pt:,}'} tok, {'found' if n.get('found') else 'MISSED'}, "
                               f"TTFT {'?' if ttft is None else f'{ttft:.0f}'}s" + (f" ({','.join(flags)})" if flags else ""))
    print("| label | " + " | ".join(f"{c} decode / prefill" for c in cases) + " | needle |")
    print("|---|" + "---:|" * len(cases) + "---|")
    for name, r in rows.items():
        print(f"| {name} | " + " | ".join(r.get(c, "-") for c in cases) + f" | {r.get('needle', '-')} |")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--table":
        return table(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--key-file", default=os.path.expanduser("~/spark-llm/.api_key"))
    ap.add_argument("--model", help="model name (default: first from /v1/models)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--cases", default="short,4k,16k,32k", help=f"comma-separated, from: {','.join(CASES)}")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--extra-body", default="{}", help="JSON merged into each request (thinking switches etc.)")
    ap.add_argument("--meta", default="{}", help="JSON object describing the engine config, recorded in the result")
    ap.add_argument("--needle", type=int, default=0, help="run one long-context needle check of N tokens instead")
    ap.add_argument("--parity", action="store_true",
                    help="greedy answers to fixed prompts (64 tokens each), saved to --out: compare two engines on one model")
    ap.add_argument("--timeout", type=float, default=7200)
    ap.add_argument("--out")
    a = ap.parse_args()
    # Everything that can be wrong with the arguments is caught here, before any request.
    cases = [c.strip().lower() for c in a.cases.split(",") if c.strip()]
    unknown = [c for c in cases if c not in CASES]
    if not (a.parity or a.needle) and (unknown or not cases):
        ap.error(f"unknown case(s) {','.join(unknown) or '(none given)'}; known: {','.join(CASES)}")
    extra, meta = _json_or_none(a.extra_body), _json_or_none(a.meta)
    if not isinstance(extra, dict):
        ap.error("--extra-body must be a JSON object")
    if not isinstance(meta, dict):
        ap.error("--meta must be a JSON object")
    kf = os.path.expanduser(a.key_file)
    key = open(kf).read().strip() if os.path.exists(kf) else "none"
    model = a.model
    if not model:
        conn, u = connect(a.base_url, 30)
        conn.request("GET", u.path.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
        resp = conn.getresponse()
        raw = resp.read()
        if resp.status != 200:
            sys.exit(f"GET {a.base_url.rstrip('/')}/models: HTTP {resp.status}: {raw[:300]!r}")
        j = _json_or_none(raw)
        listed = (j.get("data") if isinstance(j, dict) else None) or []
        if not listed or not isinstance(listed[0], dict) or "id" not in listed[0]:
            sys.exit(f"GET {a.base_url.rstrip('/')}/models listed no model: {raw[:300]!r}")
        model = listed[0]["id"]
    result = {"label": a.label, "model": model, "base_url": a.base_url, "extra_body": extra, "meta": meta,
              "max_tokens": a.max_tokens, "when": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cases": {}}
    flagged = []

    if a.parity:
        result["parity"] = []
        for i, p in enumerate(PARITY_PROMPTS):
            r = stream_chat(a.base_url, key, model, [{"role": "user", "content": p}], 64, extra, a.timeout)
            result["parity"].append({"prompt": p, "text": r["text"], "completion_tokens": r["completion_tokens"],
                                     "flags": r["flags"]})
            print(f"  [{i}] {r['text'][:100]!r}" + (f"  [flagged: {','.join(r['flags'])}]" if r["flags"] else ""))
            flagged += [f"parity [{i}]: {','.join(r['flags'])}"] if r["flags"] else []
    elif a.needle:
        code, msgs = needle_messages(a.needle, uuid.uuid4().hex)
        r = stream_chat(a.base_url, key, model, msgs, 64, extra, a.timeout)
        r["found"] = code in r["text"]
        r["expected"] = code
        result["needle"] = dict(r)
        print(f"{a.label} needle {a.needle}: prompt {r['prompt_tokens']} tok, TTFT {r['ttft_s']:.1f}s, "
              f"prefill {fmt_rate(r['prefill_tps'], 0)} tok/s, found={r['found']} answer={r['text'][:60]!r}"
              + (f"  [flagged: {','.join(r['flags'])}]" if r["flags"] else ""))
        flagged += [f"needle: {','.join(r['flags'])}"] if r["flags"] else []
    else:
        w = stream_chat(a.base_url, key, model, messages_for(0, "warmup"), 32, extra, a.timeout)   # not recorded
        if w["flags"]:
            warn(f"the warm-up request was flagged ({','.join(w['flags'])}): is thinking still on, or is usage "
                 "missing? Flagged runs are left out of the medians.")
        for case in cases:
            runs = []
            for i in range(a.runs):
                r = stream_chat(a.base_url, key, model, messages_for(CASES[case], uuid.uuid4().hex),
                                a.max_tokens, extra, a.timeout)
                r.pop("text")
                runs.append(r)
                print(f"  {a.label} {case} run {i+1}: prompt {r['prompt_tokens']} tok, TTFT {r['ttft_s']:.2f}s, "
                      f"prefill {fmt_rate(r['prefill_tps'], 0)} tok/s, decode {fmt_rate(r['decode_tps'], 1)} tok/s "
                      f"({r['completion_tokens']} tok, usage={'yes' if r['usage_reported'] else 'NO'})"
                      + (f"  [flagged: {','.join(r['flags'])}; excluded from the medians]" if r["flags"] else ""),
                      flush=True)
                flagged += [f"{case} run {i+1}: {','.join(r['flags'])}"] if r["flags"] else []
            good = [x for x in runs if not x["flags"]]
            med = lambda k: statistics.median([x[k] for x in good if x[k] is not None]) if any(x[k] is not None for x in good) else None
            result["cases"][case] = {"runs": runs, "runs_used": len(good),
                                     "flags": sorted({f for x in runs for f in x["flags"]}),
                                     "median_prefill_tps": med("prefill_tps"), "median_decode_tps": med("decode_tps"),
                                     "median_ttft_s": med("ttft_s")}
        print(f"\n{a.label} ({model}) medians over unflagged runs:")
        for case, c in result["cases"].items():
            note = f"   ({c['runs_used']} of {len(c['runs'])} runs; flagged: {','.join(c['flags'])})" if c["flags"] else ""
            print(f"  {case:>5}: prefill {fmt_rate(c['median_prefill_tps'], 0):>8} tok/s   "
                  f"decode {fmt_rate(c['median_decode_tps'], 1):>6} tok/s{note}")
    if flagged:
        warn(f"{len(flagged)} run(s) flagged, see the module docstring: " + "; ".join(flagged))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
