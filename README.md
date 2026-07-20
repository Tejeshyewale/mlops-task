# MLOps Task 0 — Batch Signal Pipeline

A minimal, reproducible, observable batch job that reads OHLCV data, computes a
rolling mean on `close`, generates a binary trading signal, and writes
structured metrics + logs.

## Files

- `run.py` — main batch job
- `config.yaml` — job config (seed, window, version)
- `data.csv` — sample OHLCV dataset (10,000 rows)
- `requirements.txt` — Python dependencies
- `Dockerfile` — container build
- `metrics.json` — sample output from a successful run
- `run.log` — sample log from a successful run

## Local run

```bash
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt

python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```

No paths are hard-coded — all inputs/outputs are passed via CLI flags.

## Docker

Build:

```bash
docker build -t mlops-task .
```

Run:

```bash
docker run --rm mlops-task
```

The container has `data.csv` and `config.yaml` baked in, runs the pipeline
with the exact required CLI command internally, prints the final metrics JSON
to stdout, and exits `0` on success / non-zero on failure.

To pull `metrics.json`/`run.log` out of the container instead of just stdout:

```bash
docker run --rm -v "$(pwd)/out:/app/out" mlops-task \
  python run.py --input data.csv --config config.yaml --output out/metrics.json --log-file out/run.log
```

## Config (`config.yaml`)

```yaml
seed: 42
window: 5
version: "v1"
```

- `seed` — used to seed `numpy.random` for reproducibility
- `window` — rolling window size (in rows) applied to `close`
- `version` — tagged into every metrics output

## Processing logic

1. Load and validate `config.yaml` (must contain `seed`, `window`, `version`).
2. Seed NumPy's RNG with `seed`.
3. Load `data.csv`, validate it's non-empty, readable, and contains a `close`
   column (numeric).
4. Compute `rolling_mean = close.rolling(window=window, min_periods=window).mean()`.
   The first `window - 1` rows have no full window yet, so their rolling mean
   is `NaN` — these rows are treated as **not eligible** for a signal and are
   assigned `signal = 0` (consistent, documented default rather than dropped
   rows, so `rows_processed` always equals the input row count).
5. `signal = 1` if `close > rolling_mean`, else `0`.
6. Compute metrics: `rows_processed`, `signal_rate = mean(signal)`, and
   `latency_ms` (total wall-clock runtime of the job).

## Output: `metrics.json`

### Success

```json
{
  "version": "v1",
  "rows_processed": 10000,
  "metric": "signal_rate",
  "value": 0.4989,
  "latency_ms": 65,
  "seed": 42,
  "status": "success"
}
```

### Error

```json
{
  "version": "v1",
  "status": "error",
  "error_message": "Description of what went wrong"
}
```

`metrics.json` is **always** written, in both success and error paths.

## Logging (`run.log`)

Every run logs (to both `run.log` and stderr):

- Job start timestamp
- Loaded/validated config (seed, window, version)
- Rows loaded
- Processing steps (rolling mean, signal generation)
- Metrics summary
- Job end + final status
- Full exception traceback on any validation/runtime error

## Error handling covered

- Missing input file
- Empty input file
- Invalid/corrupt CSV
- Missing `close` column
- Non-numeric `close` column
- Missing config file
- Invalid YAML syntax
- Missing required config fields (`seed`, `window`, `version`)
- Invalid config field types (e.g. non-integer `window`)

In every failure case, `run.py` exits with a **non-zero** status code and
still writes a valid `metrics.json` with `"status": "error"`.

## Reproducibility

Given the same `data.csv` + `config.yaml`, `rows_processed`, `signal_rate`,
and `seed` are identical across runs (only `latency_ms` varies, since it
measures wall-clock time). NumPy's RNG is seeded from `config.seed` for any
future stochastic steps. !!!

Make a Star ⭐ if you help the Repo  :) !!
