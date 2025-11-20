#!/usr/bin/env python3
"""Check threshold values in W&B runs."""
import wandb
import sys
from collections import defaultdict

if len(sys.argv) < 4:
    print("Usage: python check_thresholds.py <entity> <project> <group>")
    sys.exit(1)

entity = sys.argv[1]
project = sys.argv[2]
group = sys.argv[3]

api = wandb.Api()
runs = list(api.runs(f"{entity}/{project}", filters={"group": group}, per_page=300))
finished = [r for r in runs if r.state == "finished"]

print(f"Found {len(runs)} runs in group '{group}', {len(finished)} finished\n")

# Group runs by threshold value
threshold_groups = defaultdict(list)

for run in finished[:30]:  # Check first 30 runs
    config = getattr(run, "config", {})

    # Look for threshold-related config keys
    threshold = None
    for key in config.keys():
        if 'threshold' in key.lower() or 'tau' in key.lower():
            threshold = config.get(key)
            print(f"Run: {run.name:50s} | {key}: {threshold}")
            threshold_groups[threshold].append(run.name)
            break

    if threshold is None:
        # Check summary for threshold values
        summary = getattr(run, "summary", {})
        for key in summary.keys():
            if 'threshold' in key.lower() or 'tau' in key.lower():
                threshold = summary.get(key)
                print(f"Run: {run.name:50s} | {key} (summary): {threshold}")
                threshold_groups[threshold].append(run.name)
                break

    if threshold is None:
        print(f"Run: {run.name:50s} | No threshold found in config or summary")

print("\n" + "="*80)
print("Summary by threshold value:")
print("="*80)
for threshold, names in sorted(threshold_groups.items()):
    print(f"\nThreshold = {threshold} ({len(names)} runs):")
    for name in names[:5]:
        print(f"  - {name}")
    if len(names) > 5:
        print(f"  ... and {len(names) - 5} more")
