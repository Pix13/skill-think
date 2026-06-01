#!/usr/bin/env python3
"""
Smoke test: confirm thinking can be switched ON (and OFF) on a live server.

Reuses think-chat.py's body builder + sender, so it tests the real code path.
Sends the same prompt twice — thinking on, then off — and asserts that ON
produces a reasoning trace while OFF does not.

Usage:
    ./test_thinking.py                              # defaults to luna on jarvis02
    URL=http://host:8080 MODEL=mymodel ./test_thinking.py
    ./test_thinking.py --backend vllm --url http://host:8000 --model Qwen/Qwen3-8B

Exit code 0 = pass, 1 = fail, 2 = server unreachable (skipped).
"""
import argparse
import importlib.util
import os
import sys
import urllib.error
from pathlib import Path

# import the hyphenated think-chat.py as a module
_spec = importlib.util.spec_from_file_location(
    "think_chat", Path(__file__).parent / "think-chat.py"
)
tc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tc)


def reasoning_of(resp):
    """Pull the reasoning trace out of a response, tolerating both shapes:
    a dedicated reasoning_content field, or an inline <think>...</think> block."""
    msg = resp["choices"][0]["message"]
    rc = (msg.get("reasoning_content") or "").strip()
    if rc:
        return rc
    content = msg.get("content") or ""
    if "<think>" in content:
        return content.split("<think>", 1)[1].split("</think>", 1)[0].strip()
    return ""


def run(args, thinking):
    """Build a request via think-chat's own builder and send it."""
    ns = argparse.Namespace(
        backend=args.backend, model=args.model, thinking=thinking,
        depth="normal", think_style="None", present="None", sampling="Default",
        system="", depth_hint=False, max_tokens=256,
    )
    body = tc.build_body(ns, "What is 17*23? Reason it out step by step.")
    return tc.send(args.url, body, timeout=120.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["llama", "vllm"], default="llama")
    ap.add_argument("--url", default=os.environ.get("URL", "http://jarvis02.dmz:8080"))
    ap.add_argument("--model", default=os.environ.get("MODEL", "luna"))
    args = ap.parse_args()

    print(f"Testing {args.model} at {args.url} ({args.backend})")
    try:
        on = run(args, "on")
        off = run(args, "off")
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"SKIP: server unreachable ({e})")
        return 2

    on_trace = reasoning_of(on)
    off_trace = reasoning_of(off)

    ok = True
    if on_trace:
        print(f"PASS: thinking ON  → reasoning trace present ({len(on_trace)} chars)")
    else:
        print("FAIL: thinking ON  → no reasoning trace (toggle not honored?)")
        ok = False
    if not off_trace:
        print("PASS: thinking OFF → no reasoning trace")
    else:
        print(f"FAIL: thinking OFF → unexpected reasoning trace ({len(off_trace)} chars)")
        ok = False

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
