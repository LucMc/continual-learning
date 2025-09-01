#!/usr/bin/env python3
import tyro
import numpy as np
import altair as alt
from pathlib import Path
from wandb_utils import fetch_runs # Assumes you have this utility function
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


def fetch_and_process_data(entity: str, project: str, group: str, metric: str) -> pd.DataFrame:
    runs = fetch_runs(entity, project, {"group": group})
    algo_data = defaultdict(lambda: defaultdict(list))
    
    for run in runs:
        history = run.history(keys=[metric, "_step"])
        algo_name = ''.join(run.name.split('_')[:-1])

        for _, row in history.iterrows():
            step = row.get("_step")
            value = row.get(metric)
            if pd.notna(step) and pd.notna(value):
                binned_step = int(round(step / 10_000) * 10_000)
                algo_data[algo_name][binned_step].append(value)
    
    data = []
    for algo_name, step_dict in algo_data.items():
        for step, values_list in step_dict.items():
            values = np.array(values_list)
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
                'n_seeds': len(values),
            })
    
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df = df.sort_values(['algorithm', 'step'])
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

    # NEW: Apply a rolling average transformation for smoothing
    # We apply this to the base chart so both the line and area are smoothed.
    smoothed_base = base.transform_window(
        # The frame defines the window size. [-10, 10] averages each point
        # with the 10 points before and 10 after it.
        frame=[-10, 10],
        # Group by algorithm to smooth each line independently.
        groupby=['algorithm'],
        # Define the new, smoothed columns.
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
            alt.Tooltip('n_seeds:Q', title='Seeds')
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
    save_html: bool = False
):
    print(f"Fetching data for group: '{group}', metric: '{metric}'")
    df = fetch_and_process_data(wandb_entity, wandb_project, group, metric)
    
    if df.empty:
        print(f"No data found for group: '{group}', metric: '{metric}'. Please check your inputs.")
        return
    
    print("\nData summary:")
    for algo in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algo]
        n_seeds = algo_df['n_seeds'].iloc[0] if not algo_df.empty else 0
        print(f"  - {algo}: {n_seeds} seeds, {len(algo_df)} aggregated data points")
    
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
