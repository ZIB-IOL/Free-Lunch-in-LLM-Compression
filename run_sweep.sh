#!/usr/bin/env bash
set -euo pipefail

sweep_id="your-entity/your-project/your-sweep-id"
num_runs="${1:-1}"

wandb agent --count "$num_runs" "$sweep_id"