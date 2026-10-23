#!/usr/bin/env bash
# Build, run the tests, fly all 63 missions and plot the results.
set -euo pipefail
cd "$(dirname "$0")"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure

python3 scripts/run_missions.py "$@"

if [ "$#" -gt 0 ]; then
    echo "subset flown; skipping plot"
elif python3 -c "import matplotlib" 2>/dev/null; then
    python3 scripts/plot_results.py
else
    echo "matplotlib not installed; skipping plot (pip install -r requirements.txt)"
fi
