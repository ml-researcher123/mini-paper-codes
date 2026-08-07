#!/usr/bin/env python3
"""Git askpass helper: emits credentials from memory, never from a tracked file."""

import os
import sys


prompt = " ".join(sys.argv[1:]).lower()
if "username" in prompt:
    print("x-access-token")
else:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        raise SystemExit("GITHUB_TOKEN is not set")
    print(token)
