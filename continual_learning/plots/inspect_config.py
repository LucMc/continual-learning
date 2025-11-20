#!/usr/bin/env python3
"""Inspect config keys in W&B runs."""
import wandb
import sys

if len(sys.argv) < 4:
    print("Usage: python inspect_config.py <entity> <project> <group>")
    sys.exit(1)

entity = sys.argv[1]
project = sys.argv[2]
group = sys.argv[3]

api = wandb.Api()
runs = list(api.runs(f"{entity}/{project}", filters={"group": group}, per_page=300))
finished = [r for r in runs if r.state == "finished"]

print(f"Found {len(runs)} runs in group '{group}', {len(finished)} finished\n")

# Check first run from each pattern
patterns_checked = set()

for run in finished:
    # Group runs by prefix
    name = run.name
    if name.startswith("ccbp_br_adam") and "smaller_net" not in patterns_checked:
        patterns_checked.add("smaller_net")
        print(f"\n{'='*80}")
        print(f"Run: {name}")
        print(f"{'='*80}")
        config = getattr(run, "config", {})
        print("\nAll config keys:")
        for key in sorted(config.keys()):
            value = config.get(key)
            print(f"  {key}: {value}")
    elif name.startswith("ccbp_bigger") and "bigger" not in patterns_checked:
        patterns_checked.add("bigger")
        print(f"\n{'='*80}")
        print(f"Run: {name}")
        print(f"{'='*80}")
        config = getattr(run, "config", {})
        print("\nAll config keys:")
        for key in sorted(config.keys()):
            value = config.get(key)
            print(f"  {key}: {value}")
    elif name.startswith("ccbp_smaller") and "ccbp_smaller" not in patterns_checked:
        patterns_checked.add("ccbp_smaller")
        print(f"\n{'='*80}")
        print(f"Run: {name}")
        print(f"{'='*80}")
        config = getattr(run, "config", {})
        print("\nAll config keys:")
        for key in sorted(config.keys()):
            value = config.get(key)
            print(f"  {key}: {value}")
    elif name.startswith("ccbp_s") and "ccbp_s" not in patterns_checked:
        patterns_checked.add("ccbp_s")
        print(f"\n{'='*80}")
        print(f"Run: {name}")
        print(f"{'='*80}")
        config = getattr(run, "config", {})
        print("\nAll config keys:")
        for key in sorted(config.keys()):
            value = config.get(key)
            print(f"  {key}: {value}")

    if len(patterns_checked) >= 2:
        break
