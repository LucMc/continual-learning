#!/usr/bin/env python3
import tyro
import numpy as np
import altair as alt
from pathlib import Path
import wandb  # Add direct wandb import
import pandas as pd
from collections import defaultdict


def compute_iqm(values: np.ndarray) -> float:
    if len(values) == 0:
        return np.nan

    if len(values) < 4:
        return np.mean(values)
    
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    
    if np.any(mask):
        return np.mean(values[mask])

    return np.mean(values)


def fetch_runs(entity: str, project: str, group: str):
    api = wandb.Api()
    
    filters = {"group": group}
    
    runs = api.runs(
        f"{entity}/{project}",
        filters=filters,
        per_page=300
    )
    
    all_runs = list(runs)
    
    print(f"Total runs fetched: {len(all_runs)}")
    
    run_states = defaultdict(int)
    for run in all_runs:
        run_states[run.state] += 1
    print(f"Run states: {dict(run_states)}")
    
    return [x for x in all_runs if x.state == "finished"] # Only finished runs


def fetch_and_process_data(entity: str, project: str, group: str, metric: str) -> pd.DataFrame:
    runs = fetch_runs(entity, project, group)
    
    algo_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    runs_per_algo = defaultdict(set)
    
    for run in runs:
        parts = run.name.split('_')
        if len(parts) >= 2:
            algo_name = '_'.join(parts[:-1])
            seed_id = parts[-1]
        else:
            algo_name = run.name
            seed_id = '0'
        
        runs_per_algo[algo_name].add(seed_id)
        
        # Try to get history with the metric
        history = run.history(keys=[metric, "_step"])
        assert not history.empty, f"Warning: Run {run.name} has no data for metric '{metric}'"
        
        data_points = 0
        for _, row in history.iterrows():
            step = row.get("_step")
            value = row.get(metric)
            if pd.notna(step) and pd.notna(value):
                binned_step = int(round(step / 10_000) * 10_000)
                algo_data[algo_name][binned_step][seed_id].append(value)
                data_points += 1
        
        assert data_points > 0, f"Warning: Run {run.name} has metric but no valid data points"
        print(f"  Run {run.name}: {data_points} data points")
    
    print("\nUnique seeds per algorithm:")
    for algo, seeds in runs_per_algo.items():
        print(f"{algo}: {len(seeds)} seeds: {sorted(seeds)[:15]}{'>15' if len(seeds) > 15 else ''}")
    
    data = []
    for algo_name, step_dict in algo_data.items():
        for step, seed_dict in step_dict.items():
            # Flatten all values from all seeds at this step
            all_values = []
            for seed_values in seed_dict.values():
                all_values.extend(seed_values)
            
            if all_values:
                values = np.array(all_values)
                iqm_value = compute_iqm(values)
                
                if len(values) > 1:
                    q25, q75 = np.percentile(values, [25, 75])
                else:
                    q25, q75 = iqm_value, iqm_value

                data.append({
                    'algorithm': algo_name,
                    'step': step / 1_000_000,
                    'iqm': iqm_value,
                    'q25': q25,
                    'q75': q75,
                    'n_seeds': len(seed_dict),  # Count unique seeds at this step
                    'n_values': len(all_values),  # Total values across seeds
                })
    
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.sort_values(['algorithm', 'step'])
    
    # Print more detailed summary
    print("\nData summary:")
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        max_seeds = algo_df['n_seeds'].max()
        total_points = len(algo_df)
        print(f"  - {algo}: up to {max_seeds} seeds, {total_points} aggregated data points")
    
    return df


def create_chart(df: pd.DataFrame, metric: str, title: str = "") -> alt.Chart:
    """Create a smoothed Altair chart with a line for IQM and a shaded area for IQR."""
    
    base = alt.Chart(df).encode(
        x=alt.X('step:Q', 
                title='Training Steps (Millions)',
                scale=alt.Scale(domain=[0, 400], nice=True)),
        color=alt.Color('algorithm:N', 
                        title='Algorithm',
                        legend=alt.Legend(
                            title=None,
                            symbolOpacity=1.0,
                            orient='bottom-left',
                            fillColor='rgba(255,255,255,1)',
                            strokeColor='gray',
                            padding=5,
                            cornerRadius=3
                        ))
    )

    # Apply a rolling average transformation for smoothing
    smoothed_base = base.transform_window(
        frame=[-10, 10],
        groupby=['algorithm'],
        smooth_iqm='mean(iqm)',
        smooth_q25='mean(q25)',
        smooth_q75='mean(q75)'
    )

    # Create the shaded area using the SMOOTHED q25 and q75 values
    bands = smoothed_base.mark_area(opacity=0.25).encode(
        y=alt.Y('smooth_q25:Q', title=metric.replace('_', ' ').title()),
        y2=alt.Y2('smooth_q75:Q', title=""),
    )

    # Create the line chart using the SMOOTHED iqm value
    lines = smoothed_base.mark_line(strokeWidth=2).encode(
        y=alt.Y('smooth_iqm:Q', title=""),
        tooltip=[
            alt.Tooltip('algorithm:N', title='Algorithm'),
            alt.Tooltip('step:Q', title='Step (M)', format='.2f'),
            alt.Tooltip('smooth_iqm:Q', title='Smoothed IQM', format='.2f'),
            alt.Tooltip('n_seeds:Q', title='Seeds'),
            alt.Tooltip('n_values:Q', title='Total Values')
        ]
    )
    
    chart = bands + lines
    
    return chart.properties(
        width=1000,
        height=400,
        title=title if title else f"{metric.replace('_', ' ').title()} (IQM over Seeds)"
    ).configure_axis(
        grid=True,
        gridOpacity=0.3
    ).interactive()


def main(
    wandb_entity: str,
    wandb_project: str = "crl_experiments",
    group: str = "default_group",
    metric: str = "eval_loss",
    output_dir: str = "./plots",
    save_html: bool = False,
    debug: bool = False
):
    if debug:
        # Set wandb to debug mode for more verbose output
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    print(f"Fetching data for group: '{group}', metric: '{metric}'")
    df = fetch_and_process_data(wandb_entity, wandb_project, group, metric)
    
    if df.empty:
        print(f"No data found for group: '{group}', metric: '{metric}'. Please check your inputs.")
        print("\nTroubleshooting tips:")
        print("1. Verify the metric name is exact (check W&B UI)")
        print("2. Ensure runs in this group have logged this metric")
        print("3. Check if runs are still running/crashed")
        return
    
    title = f"{group.replace('_', ' ').title()}: {metric.replace('_', ' ').title()} (IQM)"
    chart = create_chart(df, metric, title)
    
    ext = "html" if save_html else "svg"
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    save_path = Path(output_dir) / f"{group}_{metric.replace('/', '_')}_iqm_smoothed.{ext}"
    chart.save(str(save_path))
    print(f"\n✅ Chart saved to: {save_path}")
    
    return chart


if __name__ == "__main__":
    tyro.cli(main)
