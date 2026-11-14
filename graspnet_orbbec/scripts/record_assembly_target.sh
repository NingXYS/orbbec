#!/usr/bin/env bash
set -euo pipefail
assembly_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$assembly_root:$assembly_root/../agilex_official_ws/python"
exec "$assembly_root/.conda/bin/python" "$assembly_root/scripts/record_assembly_target.py" "$@"
