#!/usr/bin/env python3
"""Quick script to check what runs exist in a W&B group."""
import wandb
import sys

if len(sys.argv) < 4:
    print("Usage: python check_runs.py <entity> <project> <group>")
    sys.exit(1)

entity = sys.argv[1]
project = sys.argv[2]
group = sys.argv[3]

api = wandb.Api()
runs = list(api.runs(f"{entity}/{project}", filters={"group": group}, per_page=300))

print(f"Found {len(runs)} runs in group '{group}'")
print(f"Finished runs: {sum(1 for r in runs if r.state == 'finished')}")
print("\nRun names:")
for r in runs[:30]:  # Show first 30
    print(f"  - {r.name} (state: {r.state})")
if len(runs) > 30:
    print(f"  ... and {len(runs) - 30} more")
