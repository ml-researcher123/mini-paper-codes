"""Paste this entire file into one Google Colab notebook cell and run it."""

import os
import subprocess
import sys
from pathlib import Path

from google.colab import userdata


REPOSITORY = "https://github.com/ml-researcher123/mini-paper-codes.git"
BRANCH = "main"
CHECKOUT = Path("/content/mini-paper-codes")

try:
    token = userdata.get("GITHUB_TOKEN")
except Exception as exc:
    raise RuntimeError(
        "Add GITHUB_TOKEN under Colab's Secrets (key icon), enable notebook access, and rerun."
    ) from exc
if not token:
    raise RuntimeError(
        "Colab secret GITHUB_TOKEN is missing or notebook access is disabled."
    )
os.environ["GITHUB_TOKEN"] = token

if not (CHECKOUT / ".git").exists():
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, REPOSITORY, str(CHECKOUT)],
        check=True,
    )
else:
    subprocess.run(
        ["git", "pull", "--ff-only", "origin", BRANCH],
        cwd=CHECKOUT,
        check=True,
    )

# HTTP/1.1 is more reliable for long-lived notebook workers and GitHub pushes.
subprocess.run(
    ["git", "config", "http.version", "HTTP/1.1"],
    cwd=CHECKOUT,
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", str(CHECKOUT / "requirements-kaggle.txt")],
    check=True,
)

# On a fresh Colab runtime, kaggle_worker.py seeds its work directory from the
# latest results/<job_id> checkpoint in GitHub and skips every completed ID.
subprocess.run(
    [
        sys.executable,
        str(CHECKOUT / "kaggle_worker.py"),
        "--repo-dir",
        str(CHECKOUT),
        "--branch",
        BRANCH,
        "--poll-seconds",
        "60",
        "--checkpoint-minutes",
        "10",
    ],
    check=True,
    env=os.environ.copy(),
)
