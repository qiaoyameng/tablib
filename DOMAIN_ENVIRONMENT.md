# Reproducible offline environment

Upstream: https://github.com/jazzband/tablib
Upstream SHA: b6a75dcbe01cc42b3fd45ae5005901e2639fa7a7
Runtime: 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]
Platform: win32

Create an isolated virtual environment, then run from this checkout:
```
python -m pip install -r requirements-domain.lock
python -m pip install -e .
```
Set `PYTHONUTF8=1` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` before tests.
```
python -m pytest -q -o addopts= -o log_cli=false tests/test_tablib.py tests/test_registry_import.py
```
Measured selected pytest process: 1.759 seconds.
No pandas dependency is installed. Optional spreadsheet formats are test-only dependencies; the proposed core tasks operate on Dataset/CSV/JSON.

Tests use synthetic in-memory or temporary-file inputs under the upstream license.
No account, cloud service, browser or external database is needed. Recreate the
environment from this immutable commit to reset the starting state.
