from argparse import ArgumentParser
import os

import numpy as np
import pandas as pd
import wandb


def main(args):

    if args.names is not None:
        if args.runs is None:
            raise Exception("Renaming copied runs not supported when copying whole project.")
        assert len(args.names) == len(args.runs), "Number of new names must equal number of run IDs"

    args.dst_entity = args.dst_entity if args.dst_entity is not None else args.src_entity
    args.dst_project = args.dst_project if args.dst_project is not None else args.src_project
    same_project = args.src_entity == args.dst_entity and args.src_project == args.dst_project
    if same_project and args.names is None:
        name_append = "-copy"
    else:
        name_append = ""

    # Prepare dedicated directories to avoid clutter
    base_dir = args.out_dir if args.out_dir is not None else os.path.join(os.getcwd(), "wandb_copy_runs")
    cache_dir = os.path.join(base_dir, "cache")
    downloads_root = os.path.join(base_dir, "downloads")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(downloads_root, exist_ok=True)

    # Ensure W&B uses the dedicated directory for run files/cache
    os.environ["WANDB_DIR"] = base_dir
    os.environ["WANDB_CACHE_DIR"] = cache_dir

    # Set your API key
    wandb.login()
    # Initialize the wandb API
    api = wandb.Api()

    # Get the runs from the source project (optionally filter by group)
    filters = {}
    if args.src_group is not None:
        filters["group"] = args.src_group
    runs = api.runs(f"{args.src_entity}/{args.src_project}", filters=filters if filters else None)

    # Iterate through the runs and copy them to the destination project
    for run in runs:
        if args.runs is not None and run.id not in args.runs:
            continue
        # Get the run history and files
        last_step = getattr(run, "lastHistoryStep", None)
        samples = (last_step + 1) if isinstance(last_step, int) else None
        history = run.history(samples=samples) if samples is not None else run.history()
        system = run.history(samples=samples, stream="system") if samples is not None else run.history(stream="system")
        history = history.join(system, rsuffix="_system")
        files = run.files()

        name = run.name if args.names is None else args.names[args.runs.index(run.id)]

        # Create a new run in the destination project

        # Log the history to the new run
        new_run = wandb.init(
            project=args.dst_project,
            entity=args.dst_entity,
            config=run.config,
            name=name + name_append,
            group=(args.dst_group if args.dst_group is not None else run.group),
            resume="allow",
            dir=base_dir,
        )
        for index, row in history.iterrows():
            # Keep None values as in original, skip NaNs robustly
            row_dict = row.to_dict()

            # Extract original step if present to preserve x-axis alignment
            step_val = None
            if "_step" in row_dict:
                try:
                    if not pd.isna(row_dict["_step"]):
                        step_val = int(row_dict["_step"])  # preserve original _step
                except Exception:
                    step_val = None

            # Build payload, excluding special _step key (handled via step=)
            payload = {}
            for k, v in row_dict.items():
                if k == "_step":
                    continue
                try:
                    is_nan = (v == "NaN") or pd.isna(v)
                except Exception:
                    is_nan = False
                if v is None or not is_nan:
                    payload[k] = v

            # Log with preserved step when available
            if step_val is not None:
                new_run.log(payload, step=step_val)
            else:
                new_run.log(payload)

        # Upload the files to the new run
        # Download files to a scoped path and save from there
        run_download_dir = os.path.join(downloads_root, run.id)
        os.makedirs(run_download_dir, exist_ok=True)
        for file in files:
            file.download(root=run_download_dir, replace=True)
            new_run.save(os.path.join(run_download_dir, file.name), policy="now")

        # Finish the new run
        new_run.finish()


if __name__ == "__main__":
    parser = ArgumentParser(description="Copies one or all of the runs in a wandb project to another.")
    parser.add_argument("-se", "--src-entity", type=str, default=None, help="Source wandb entity name.")
    parser.add_argument("-sp", "--src-project", type=str, help="Name of the wandb projecet.")
    parser.add_argument("-de", "--dst-entity", type=str, default=None, help="Destination wandb entity name.")
    parser.add_argument("-dp", "--dst-project", type=str, default=None, help="Name of destination wandb project.")
    parser.add_argument("--src-group", type=str, default=None, help="Optional: Source group name to filter runs.")
    parser.add_argument("--dst-group", type=str, default=None, help="Optional: Destination group name for copied runs.")
    parser.add_argument("-r", "--runs", nargs="*", type=str, default=None, help="List of run IDs to copy. If None will copy all in project.")
    parser.add_argument("-n", "--names", nargs="*", type=str, default=None, help="List of new names for copied runs (optional).")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory to store local W&B files and downloads (reduces clutter). Defaults to ./wandb_copy_runs.")

    main(parser.parse_args())
