"""Paste this entire file into one Kaggle notebook cell and run it."""

import os
import subprocess
import sys
from pathlib import Path

from kaggle_secrets import UserSecretsClient

REPOSITORY = "https://github.com/ml-researcher123/mini-paper-codes.git"
BRANCH = "main"
CHECKOUT = Path("/kaggle/working/mini-paper-codes")

token = UserSecretsClient().get_secret("GITHUB_TOKEN")
if not token:
    raise RuntimeError("Kaggle secret GITHUB_TOKEN is missing or not enabled.")
os.environ["GITHUB_TOKEN"] = token

if not (CHECKOUT / ".git").exists():
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, REPOSITORY, str(CHECKOUT)],
        check=True,
    )
else:
    # Pull dependency pins and job changes before installing requirements.
    subprocess.run(
        ["git", "pull", "--ff-only", "origin", BRANCH],
        cwd=CHECKOUT,
        check=True,
    )

subprocess.run(
    [sys.executable, "-m", "pip", "install", "--upgrade", "-q", "-r", str(CHECKOUT / "requirements-kaggle.txt")],
    check=True,
)
# This call intentionally blocks while the notebook is acting as a worker.
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
        "20",
    ],
    check=True,
    env=os.environ.copy(),
)
