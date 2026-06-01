#!/usr/bin/env python3
"""
think-chat — drive a local Qwen3-style model with reasoning toggled per request.

Talks to an OpenAI-compatible local server (llama-server or vLLM) and lets an
agent flip thinking ON/OFF, set its depth, and apply reasoning-style / sampling /
presentation presets — without editing any server config.

Stdlib only (urllib). No pip installs.

Examples
--------
  # fast answer, thinking OFF (the default)
  ./think-chat.py "what's 2+2?"

  # turn thinking ON with a deep budget for a hard subtask
  ./think-chat.py --thinking on --depth deep "prove sqrt(2) is irrational"

  # vLLM backend, show the reasoning trace too
  ./think-chat.py --backend vllm --thinking on --show-thinking "plan this refactor"

  # apply a reasoning style + sampling preset
  ./think-chat.py --thinking on --think-style "First Principles" \
                  --sampling "Thinking Precise" "design a rate limiter"

  # inspect the exact request body without sending it
  ./think-chat.py --thinking on --depth max --dry-run "hello"

Run with --list to see every preset name.
"""
import argparse
import json
import sys
import urllib.request
import urllib.error

# ── Thinking depth → token budget ───────────────────────────────────────────
# llama-server: sent as `reasoning_budget` (-1 = unlimited, 0 = off, N = cap).
# vLLM:         sent as `thinking_budget` when thinking is ON and depth != unlimited
#               (requires a vLLM build with reasoning-budget support; see SKILL.md).
DEPTH_MAP = {
    "unlimited": -1,
    "max": 16000,
    "deep": 8192,
    "normal": 3072,
    "quick": 512,
}

DEPTH_HINTS = {
    "unlimited": "",
    "max": "You have an enormous reasoning budget. Use every token of it. Explore exhaustively.",
    "deep": "Reason carefully and thoroughly through every step before answering.",
    "normal": "Think through the problem before answering, but stay concise.",
    "quick": "Think briefly, then give a direct and concise answer.",
}

# ── Reasoning-style presets (injected into the system message) ───────────────
THINKING_PRESETS = {
    "None": "",
    "Think Less": "Do not overthink. Reach a conclusion quickly and avoid excessive reasoning.",
    "Think More": "Think deeply and explore multiple angles before settling on an answer.",
    "Extended Thinking": (
        "Before answering, conduct an extended internal deliberation. "
        "Explore the problem space broadly: consider edge cases, alternative interpretations, "
        "counterarguments, and second-order effects. Stress-test your initial conclusions. "
        "Only converge on an answer after exhausting the major lines of reasoning."
    ),
    "MAX Thinking": (
        "You are in maximum reasoning mode. Think as long and as deeply as humanly possible — "
        "do not cut corners, do not rush to conclusions. "
        "Map out every relevant concept, assumption, and implication. "
        "Consider the problem from first principles, from domain expertise, from edge cases, "
        "and from adversarial angles. Challenge every intermediate conclusion before accepting it. "
        "If a chain of reasoning feels complete, ask yourself: what have I missed? "
        "Only after exhaustive internal deliberation should you begin composing your answer."
    ),
    "Step by Step": "Break down your reasoning into clear numbered steps.",
    "Devil's Advocate": "Consider and steelman the opposing viewpoint before giving your answer.",
    "First Principles": (
        "Strip the problem down to its most fundamental truths. "
        "Refuse to rely on analogy or convention. "
        "Rebuild your reasoning from the ground up using only what can be directly justified."
    ),
    "10x Hypotheses": (
        "Before answering, generate at least ten distinct hypotheses or approaches. "
        "Briefly evaluate each. Then select and develop the most promising one."
    ),
    "Socratic": (
        "Interrogate the question itself before answering. "
        "What assumptions does it carry? Are those assumptions valid? "
        "What is really being asked beneath the surface?"
    ),
    "Rubber Duck": (
        "Explain your reasoning step by step out loud as if teaching it to someone who knows nothing. "
        "Narrate every logical move. Catch your own mistakes as you speak them."
    ),
    "Pre-Mortem": (
        "Assume your answer or plan will fail. "
        "Think through every plausible reason it could go wrong before you commit to it. "
        "Then adjust your answer to address those failure modes."
    ),
    "Bayesian": (
        "Reason probabilistically. "
        "Assign rough confidence levels to key claims. "
        "Update them as you reason. "
        "State your final answer with an honest calibrated uncertainty."
    ),
    "Contrarian": (
        "Your default stance is skepticism. "
        "Challenge the framing of the question. "
        "Push back on obvious answers. "
        "Only accept a conclusion if it survives hard scrutiny."
    ),
}

# ── Presentation presets (injected into the system message) ──────────────────
PRESENTATION_PRESETS = {
    "None": "",
    "ELI5": "After thinking, explain your answer as simply as possible, like I'm five.",
    "Be Concise": "After thinking, give the shortest possible answer that is still complete.",
    "Bullet Points": "Present your final answer as a clean bulleted list.",
    "TL;DR First": "Open your answer with a one-sentence TL;DR, then elaborate.",
    "Teach Me": (
        "Present your answer as a mini-lesson: start with the core concept, "
        "build up with examples, and end with a memorable takeaway."
    ),
    "Expert Tone": (
        "Write your answer at graduate level. "
        "Use precise technical vocabulary. Assume the reader is a domain expert."
    ),
    "Casual Chat": "Write your answer like you're texting a smart friend. Relaxed, no jargon.",
    "Debate Format": (
        "Structure your answer as a formal debate: "
        "state the proposition, present the strongest case for it, "
        "then the strongest case against it, then your verdict."
    ),
    "Analogies Only": (
        "Explain everything exclusively through analogies and metaphors. "
        "Do not use technical terms — map every concept to something concrete and familiar."
    ),
    "Action Items": "Distill your answer into concrete, numbered action items the user can execute immediately.",
    "Socratic Reply": (
        "Instead of stating conclusions directly, guide the user to them "
        "through a sequence of probing questions."
    ),
}

# ── Sampling presets (root-level params; mirrored into options{} for Ollama-routed) ──
SAMPLING_PRESETS = {
    "Default": None,  # touch nothing
    "Instruct General": {
        "temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repeat_penalty": 1.0, "repetition_penalty": 1.0,
    },
    "Instruct Reasoning": {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repeat_penalty": 1.0, "repetition_penalty": 1.0,
    },
    "Thinking General": {
        "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 1.5, "repeat_penalty": 1.0, "repetition_penalty": 1.0,
    },
    "Thinking Precise": {  # coding / precise tasks
        "temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
        "presence_penalty": 0.0, "repeat_penalty": 1.0, "repetition_penalty": 1.0,
    },
}

DEFAULT_URLS = {"llama": "http://localhost:8080", "vllm": "http://localhost:8000"}


def build_body(args, prompt):
    """Construct the OpenAI-compatible chat-completions request body."""
    thinking_on = args.thinking == "on"
    depth = args.depth

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    body = {
        "model": args.model,
        "messages": messages,
        "stream": False,
    }
    if args.max_tokens is not None:
        body["max_tokens"] = args.max_tokens

    # ── thinking toggle (both backends understand chat_template_kwargs) ──
    body["chat_template_kwargs"] = {"enable_thinking": thinking_on}

    budget = DEPTH_MAP.get(depth, -1)
    if args.backend == "llama":
        # llama.cpp: always send reasoning_budget. 0 hard-disables thinking.
        body["reasoning_budget"] = budget if thinking_on else 0
    else:  # vllm
        # vLLM caps via a separate field, only meaningful when thinking is on
        # and a finite budget is requested. Unlimited (-1) => omit, let it run.
        if thinking_on and budget > 0:
            body["thinking_budget"] = budget

    # ── sampling params ──
    sampling = SAMPLING_PRESETS.get(args.sampling)
    if sampling:
        for k, v in sampling.items():
            body[k] = v
        options = body.setdefault("options", {})
        for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty"):
            if k in sampling:
                options[k] = sampling[k]

    # ── prompt-injection presets (depth hint + style + presentation) ──
    parts = []
    if thinking_on and args.depth_hint:
        h = DEPTH_HINTS.get(depth, "")
        if h:
            parts.append(h)
    if thinking_on:
        t = THINKING_PRESETS.get(args.think_style, "")
        if t:
            parts.append(t)
    p = PRESENTATION_PRESETS.get(args.present, "")
    if p:
        parts.append(p)

    if parts:
        injection = "\n\n".join(parts)
        sys_idx = next((i for i, m in enumerate(messages) if m["role"] == "system"), None)
        if sys_idx is not None:
            existing = messages[sys_idx]["content"]
            messages[sys_idx]["content"] = f"{injection}\n\n{existing}" if existing else injection
        else:
            messages.insert(0, {"role": "system", "content": injection})

    return body


def send(url, body, timeout):
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        endpoint, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main():
    ap = argparse.ArgumentParser(
        description="Drive a local Qwen3-style model with reasoning toggled per request.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("prompt", nargs="?", help="User prompt (or pipe via stdin).")
    ap.add_argument("--backend", choices=["llama", "vllm"], default="llama",
                    help="Target server (default: llama).")
    ap.add_argument("--url", help="Base URL (default: llama=:8080, vllm=:8000).")
    ap.add_argument("--model", default="local",
                    help="Model name sent to the server (default: 'local').")
    ap.add_argument("--thinking", choices=["on", "off"], default="off",
                    help="Reasoning mode (default: off — enable it per request when needed).")
    ap.add_argument("--depth", choices=list(DEPTH_MAP), default=None,
                    help="Reasoning budget when thinking is on "
                         "(default: 'deep' with --thinking on, else 'unlimited').")
    ap.add_argument("--think-style", default="None", metavar="PRESET",
                    help="Reasoning-style preset (see --list).")
    ap.add_argument("--present", default="None", metavar="PRESET",
                    help="Presentation preset (see --list).")
    ap.add_argument("--sampling", default=None, metavar="PRESET",
                    help="Sampling preset (default: 'Thinking Precise' with "
                         "--thinking on, else 'Default'). See --list.")
    ap.add_argument("--system", default="", help="Base system prompt.")
    ap.add_argument("--depth-hint", action="store_true",
                    help="Inject a prompt nudge matching the chosen depth.")
    ap.add_argument("--max-tokens", type=int, help="Cap total output tokens.")
    ap.add_argument("--show-thinking", action="store_true",
                    help="Also print the reasoning trace (reasoning_content), if any.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the request body as JSON and exit (no network).")
    ap.add_argument("--raw", action="store_true",
                    help="Print the full JSON response instead of just the message.")
    ap.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout seconds.")
    ap.add_argument("--list", action="store_true", help="List all preset names and exit.")
    args = ap.parse_args()

    # Thinking-on defaults to the general-coding profile (deep budget + precise
    # sampling); explicit --depth/--sampling override it. Thinking-off is untouched.
    if args.depth is None:
        args.depth = "deep" if args.thinking == "on" else "unlimited"
    if args.sampling is None:
        args.sampling = "Thinking Precise" if args.thinking == "on" else "Default"

    if args.list:
        print("Depth:        ", ", ".join(DEPTH_MAP))
        print("Think styles: ", ", ".join(THINKING_PRESETS))
        print("Presentation: ", ", ".join(PRESENTATION_PRESETS))
        print("Sampling:     ", ", ".join(SAMPLING_PRESETS))
        return 0

    prompt = args.prompt
    if not prompt and not sys.stdin.isatty():
        prompt = sys.stdin.read().strip()
    if not prompt:
        ap.error("no prompt given (pass an argument or pipe via stdin)")

    # validate preset names early with helpful errors
    for name, table, label in [
        (args.think_style, THINKING_PRESETS, "--think-style"),
        (args.present, PRESENTATION_PRESETS, "--present"),
        (args.sampling, SAMPLING_PRESETS, "--sampling"),
    ]:
        if name not in table:
            ap.error(f"unknown {label} preset {name!r}. Run --list to see valid names.")

    url = args.url or DEFAULT_URLS[args.backend]
    body = build_body(args, prompt)

    if args.dry_run:
        print(json.dumps({"endpoint": url.rstrip("/") + "/v1/chat/completions",
                          "body": body}, indent=2))
        return 0

    try:
        resp = send(url, body, args.timeout)
    except urllib.error.HTTPError as e:
        sys.stderr.write(f"HTTP {e.code} from {url}: {e.read().decode(errors='replace')}\n")
        return 1
    except urllib.error.URLError as e:
        sys.stderr.write(f"cannot reach {url}: {e.reason}\n")
        return 1

    if args.raw:
        print(json.dumps(resp, indent=2))
        return 0

    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError):
        print(json.dumps(resp, indent=2))
        return 1

    if args.show_thinking and msg.get("reasoning_content"):
        print("─── thinking ───")
        print(msg["reasoning_content"].strip())
        print("─── answer ───")
    print((msg.get("content") or "").strip())
    return 0


if __name__ == "__main__":
    sys.exit(main())
