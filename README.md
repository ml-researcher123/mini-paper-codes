# Mini Paper Experiments

This repository contains the reproducible experiments for **Does Quantization
Change What Agreement Means? Same-Wrong-Answer Concentration in Homogeneous LLM
Panels** and a small Kaggle worker that can rerun jobs when this repository is
updated.

## Kaggle setup

1. Create a Kaggle notebook, enable **Internet**, and select **GPU T4 x2**.
2. Create a fine-grained GitHub personal access token restricted to this
   repository with **Contents: Read and write** permission.
3. Add it to the notebook through **Add-ons -> Secrets** as `GITHUB_TOKEN`, and
   enable the secret for the notebook.
4. Paste and run the cell from [`KAGGLE_NOTEBOOK_CELL.py`](KAGGLE_NOTEBOOK_CELL.py).

The cell stays alive and polls the `main` branch. Kaggle sessions are not
permanent; after a session expires, start the cell again. Completed job IDs are
recorded in `results/_completed_jobs.json`, so restarting does not duplicate a
finished run.

## Triggering a run

Edit [`kaggle_job.json`](kaggle_job.json). Every job must have a new, unique
`job_id`; a timestamp such as `2026-08-08-pilot-v2` works well. Commit and push
the code/config changes and the new job ID together. Result-only commits do not
trigger another run.

The worker:

1. fast-forwards to the newest `main` commit;
2. notices an unseen `job_id`;
3. executes the manifest's argument list without a shell;
4. checkpoints outputs to `results/<job_id>/` and pushes them periodically;
5. writes final status and provenance, then records the job as completed.

Do not put tokens in this repository. The worker reads `GITHUB_TOKEN` only from
the Kaggle secret environment and uses a non-persistent Git credential helper.

## Local smoke test

```bash
python kaggle_worker.py --repo-dir . --once --no-push
```

For the real experiment, install [`requirements-kaggle.txt`](requirements-kaggle.txt).
The experiment saves one JSONL file per model/precision/dataset condition and a
machine-readable summary. It resumes completed conditions after interruption.
