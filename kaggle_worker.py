"""Poll a Git repository, execute unseen jobs, and push checkpointed results."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_git(repo: Path, args: list[str], *, auth: bool = False, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    if auth:
        askpass = repo / "scripts" / "git_askpass.py"
        try:
            askpass.chmod(0o700)
        except OSError:
            pass
        env.update(
            {
                "GIT_ASKPASS": str(askpass),
                "GIT_ASKPASS_REQUIRE": "force",
                "GIT_TERMINAL_PROMPT": "0",
            }
        )
    return subprocess.run(
        ["git", *args], cwd=repo, env=env, text=True, capture_output=True, check=check
    )


def sync_from_remote(repo: Path, branch: str) -> None:
    run_git(repo, ["fetch", "origin", branch])
    status = run_git(repo, ["status", "--porcelain"]).stdout.strip()
    if status:
        raise RuntimeError(f"Cannot update a dirty checkout:\n{status}")
    run_git(repo, ["merge", "--ff-only", f"origin/{branch}"])


def push_pending_commits(repo: Path, branch: str, no_push: bool) -> None:
    """Retry locally committed results that a transient network failure stranded."""
    if no_push:
        return
    ahead = int(run_git(repo, ["rev-list", "--count", f"origin/{branch}..HEAD"]).stdout.strip())
    if ahead:
        pushed = run_git(repo, ["push", "origin", f"HEAD:{branch}"], auth=True, check=False)
        if pushed.returncode != 0:
            raise RuntimeError(f"Could not push {ahead} pending commit(s):\n{pushed.stderr}")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def validate_job(job: dict[str, Any]) -> None:
    if not isinstance(job.get("job_id"), str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,100}", job["job_id"]):
        raise ValueError("job_id must use only letters, numbers, dot, underscore, or hyphen")
    command = job.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
        raise ValueError("command must be a non-empty JSON array of strings")


def snapshot_work(work_dir: Path, result_dir: Path) -> None:
    staging = result_dir.with_name(result_dir.name + ".snapshot")
    if staging.exists():
        shutil.rmtree(staging)
    if work_dir.exists():
        shutil.copytree(work_dir, staging)
    else:
        staging.mkdir(parents=True)
    if result_dir.exists():
        shutil.rmtree(result_dir)
    staging.replace(result_dir)


def commit_and_push(repo: Path, branch: str, message: str, no_push: bool) -> bool:
    run_git(repo, ["add", "results"])
    changed = run_git(repo, ["diff", "--cached", "--quiet"], check=False).returncode != 0
    if not changed:
        return False
    run_git(repo, ["commit", "-m", message])
    if no_push:
        return True

    # A maintainer may have pushed a new job while this one was running.
    for attempt in range(4):
        pushed = run_git(repo, ["push", "origin", f"HEAD:{branch}"], auth=True, check=False)
        if pushed.returncode == 0:
            return True
        run_git(repo, ["fetch", "origin", branch])
        rebased = run_git(repo, ["rebase", f"origin/{branch}"], check=False)
        if rebased.returncode != 0:
            run_git(repo, ["rebase", "--abort"], check=False)
            raise RuntimeError(f"Result push conflict:\n{rebased.stderr}")
        time.sleep(2**attempt)
    raise RuntimeError(f"Could not push results after retries:\n{pushed.stderr}")


def execute_job(repo: Path, branch: str, job: dict[str, Any], work_root: Path, checkpoint_minutes: int, no_push: bool) -> None:
    job_id = job["job_id"]
    source_sha = run_git(repo, ["rev-parse", "HEAD"]).stdout.strip()
    work_dir = work_root / job_id
    result_dir = repo / "results" / job_id
    # A new Kaggle session has an empty work disk but the Git repository may
    # contain a checkpoint from an earlier session. Seed the live work tree
    # from that checkpoint so JSONL question-level progress is resumed.
    if not work_dir.exists() and result_dir.exists():
        shutil.copytree(result_dir, work_dir)
    else:
        work_dir.mkdir(parents=True, exist_ok=True)
    resume_from = job.get("resume_from_job")
    if resume_from and not any(work_dir.glob("*.jsonl")):
        source_dir = repo / "results" / resume_from
        if not source_dir.is_dir():
            raise FileNotFoundError(f"Resume source does not exist: {source_dir}")
        for source_file in source_dir.glob("*.jsonl"):
            shutil.copy2(source_file, work_dir / source_file.name)

    command = [part.replace("{output_dir}", str(work_dir)) for part in job["command"]]
    metadata = {
        "job_id": job_id,
        "source_sha": source_sha,
        "command": command,
        "started_at": utc_now(),
        "status": "running",
    }
    atomic_json(work_dir / "worker_metadata.json", metadata)

    log_handle = (work_dir / "worker.log").open("a", encoding="utf-8", buffering=1)
    try:
        job_env = os.environ.copy()
        job_env.pop("GITHUB_TOKEN", None)
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=job_env,
        )
        interval = max(1, checkpoint_minutes) * 60
        next_checkpoint = time.monotonic() + interval
        while process.poll() is None:
            time.sleep(min(15, max(1, next_checkpoint - time.monotonic())))
            if time.monotonic() >= next_checkpoint:
                log_handle.flush()
                snapshot_work(work_dir, result_dir)
                commit_and_push(repo, branch, f"[kaggle-worker] checkpoint {job_id} [skip kaggle]", no_push)
                next_checkpoint = time.monotonic() + interval
        return_code = process.returncode
    finally:
        log_handle.close()

    metadata.update(
        {
            "finished_at": utc_now(),
            "return_code": return_code,
            "status": "succeeded" if return_code == 0 else "failed",
        }
    )
    atomic_json(work_dir / "worker_metadata.json", metadata)
    snapshot_work(work_dir, result_dir)

    registry_path = repo / "results" / "_completed_jobs.json"
    registry = load_json(registry_path, {})
    registry[job_id] = {
        "status": metadata["status"],
        "source_sha": source_sha,
        "finished_at": metadata["finished_at"],
    }
    atomic_json(registry_path, registry)
    commit_and_push(repo, branch, f"[kaggle-worker] {metadata['status']} {job_id} [skip kaggle]", no_push)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-dir", type=Path, required=True)
    parser.add_argument("--branch", default="main")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--checkpoint-minutes", type=int, default=20)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo_dir.resolve()
    work_root = (args.work_root or (repo / ".worker_work")).resolve()
    run_git(repo, ["config", "user.name", "Kaggle Experiment Worker"])
    run_git(repo, ["config", "user.email", "kaggle-worker@users.noreply.github.com"])

    while True:
        try:
            sync_from_remote(repo, args.branch)
            push_pending_commits(repo, args.branch, args.no_push)
            job = load_json(repo / "kaggle_job.json", {})
            if job.get("enabled", False):
                validate_job(job)
                completed = load_json(repo / "results" / "_completed_jobs.json", {})
                if job["job_id"] not in completed:
                    print(f"[{utc_now()}] starting {job['job_id']}", flush=True)
                    execute_job(repo, args.branch, job, work_root, args.checkpoint_minutes, args.no_push)
                else:
                    recorded_status = completed[job["job_id"]].get("status", "unknown")
                    print(
                        f"[{utc_now()}] waiting; {job['job_id']} is recorded as {recorded_status}",
                        flush=True,
                    )
            else:
                print(f"[{utc_now()}] waiting; no enabled job", flush=True)
        except Exception as exc:
            print(f"[{utc_now()}] worker error: {exc}", file=sys.stderr, flush=True)
            if args.once:
                raise
        if args.once:
            return
        time.sleep(max(10, args.poll_seconds))


if __name__ == "__main__":
    main()
