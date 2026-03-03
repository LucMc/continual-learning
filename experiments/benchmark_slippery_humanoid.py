"""Extract and compare runtimes from existing wandb runs for slippery humanoid."""
import tyro
import wandb
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class Args:
    wandb_entity: str
    wandb_project: str = "crl_experiments"
    group: str = "slippery_humanoid_full6"


def main():
    args = tyro.cli(Args)

    api = wandb.Api()
    runs = api.runs(
        f"{args.wandb_entity}/{args.wandb_project}",
        filters={"group": args.group},
    )

    runtimes = defaultdict(list)

    for run in runs:
        if run.state != "finished":
            continue

        # Extract algorithm name from run name (strip seed suffix)
        name = run.name
        parts = name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            algo = parts[0]
        else:
            algo = name

        # Get runtime from wandb summary
        runtime = run.summary.get("_runtime")
        if runtime is not None:
            runtimes[algo].append(runtime)
            print(f"{name}: {runtime:.1f}s")

    if not runtimes:
        print("No runtime data found!")
        return

    # Print summary
    print("\n" + "=" * 60)
    print("AVERAGE RUNTIME BY METHOD")
    print("=" * 60)
    print(f"{'Method':<35} {'Avg (s)':<12} {'Avg (h)':<10} {'N'}")
    print("-" * 60)

    results = []
    for algo, times in sorted(runtimes.items()):
        avg = sum(times) / len(times)
        results.append((algo, avg, len(times)))

    # Sort by runtime
    results.sort(key=lambda x: x[1])

    for algo, avg, n in results:
        print(f"{algo:<35} {avg:<12.1f} {avg/3600:<10.2f} {n}")


if __name__ == "__main__":
    main()
