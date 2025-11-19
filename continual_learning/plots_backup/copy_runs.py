from argparse import ArgumentParser
import os
import pandas as pd
import wandb

def main(args):
    if args.names and not args.runs:
        raise Exception("Renaming not supported when copying whole project.")
    if args.names:
        assert len(args.names) == len(args.runs), "Names must match run IDs"

    args.dst_entity = args.dst_entity or args.src_entity
    args.dst_project = args.dst_project or args.src_project
    same_project = args.src_entity == args.dst_entity and args.src_project == args.dst_project
    name_append = "-copy" if same_project and not args.names else ""

    base_dir = args.out_dir or os.path.join(os.getcwd(), "wandb_copy_runs")
    cache_dir = os.path.join(base_dir, "cache")
    downloads_root = os.path.join(base_dir, "downloads")
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(downloads_root, exist_ok=True)
    os.environ["WANDB_DIR"] = base_dir
    os.environ["WANDB_CACHE_DIR"] = cache_dir

    wandb.login()
    api = wandb.Api()
    filters = {"group": args.src_group} if args.src_group else None
    runs = api.runs(f"{args.src_entity}/{args.src_project}", filters=filters)

    for run in runs:
        if args.runs and run.id not in args.runs: continue

        last_step = getattr(run, "lastHistoryStep", None)
        samples = (last_step + 1) if isinstance(last_step, int) else None
        history = run.history(samples=samples) if samples else run.history()
        system = run.history(samples=samples, stream="system") if samples else run.history(stream="system")
        history = history.join(system, rsuffix="_system")
        files = run.files()
        name = args.names[args.runs.index(run.id)] if args.names else run.name

        new_run = wandb.init(
            project=args.dst_project, entity=args.dst_entity, config=run.config,
            name=name + name_append, group=args.dst_group or run.group,
            resume="allow", dir=base_dir
        )

        for _, row in history.iterrows():
            row_dict = row.to_dict()
            step_val = None
            if "_step" in row_dict and not pd.isna(row_dict["_step"]):
                try:
                    step_val = int(row_dict["_step"])
                except Exception:
                    pass

            payload = {k: v for k, v in row_dict.items() if k != "_step" and (v is None or not pd.isna(v))}
            new_run.log(payload, step=step_val) if step_val else new_run.log(payload)

        run_download_dir = os.path.join(downloads_root, run.id)
        os.makedirs(run_download_dir, exist_ok=True)
        for file in files:
            file.download(root=run_download_dir, replace=True)
            new_run.save(os.path.join(run_download_dir, file.name), policy="now")
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
