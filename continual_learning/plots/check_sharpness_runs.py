#!/usr/bin/env python3
"""Check what the actual run names look like in the sharpness sweep group."""

import wandb

api = wandb.Api()

runs = api.runs(
    "lucmc/crl_experiments",
    filters={"group": "slippery_ant_ccbp_sharpness_sweep"}
)

print("Sample run names:")
for i, run in enumerate(runs[:5]):
    print(f"{i+1}. {run.name}")
    print(f"   State: {run.state}")
    print()
