#!/usr/bin/env python3
import tyro
import numpy as np
import altair as alt
from pathlib import Path
import wandb
import pandas as pd
from collections import defaultdict
from typing import Optional, Literal

def compute_iqm(values: np.ndarray) -> float:
    if len(values) < 4: return np.mean(values)
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    return np.mean(values[mask]) if np.any(mask) else np.mean(values)


def parse_run_name(run_name: str) -> dict:
    parts = run_name.split('_')
    if len(parts) < 3: return {"algorithm": run_name, "seed": "0"}

    seed = parts[-1] if parts[-1].startswith('s') else "0"
    algorithm = parts[0]
    tag_part = '_'.join(parts[1:-1])
    params = {}

    for param_pair in tag_part.split(','):
        if '=' in param_pair:
            key, value = param_pair.split('=', 1)
            try:
                params[key] = float(value) if '.' in value or 'e' in value.lower() else int(value)
            except ValueError:
                params[key] = value

    return {"algorithm": algorithm, "seed": seed, **params}


def fetch_runs(entity: str, project: str, group: str):
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}, per_page=300))
    print(f"Fetched {len(runs)} runs, {len([r for r in runs if r.state == 'finished'])} finished")
    return [r for r in runs if r.state == "finished"]


def fetch_ablation_data(entity: str, project: str, group: str, metric: str, split_by: Optional[str] = None) -> pd.DataFrame:
    runs = fetch_runs(entity, project, group)
    run_data = []

    for run in runs:
        parsed = parse_run_name(run.name)
        history = run.history(keys=[metric, "_step"])
        if history.empty: continue

        for _, row in history.iterrows():
            step, value = row.get("_step"), row.get(metric)
            if pd.notna(step) and pd.notna(value):
                run_data.append({**parsed, "step": step, "value": value, "run_name": run.name})

    if not run_data: return pd.DataFrame()

    df = pd.DataFrame(run_data)
    if split_by and split_by in df.columns:
        df['group'] = f"{split_by}=" + df[split_by].astype(str)
    else:
        df['group'] = df['algorithm']

    return df


def create_short_labels(df: pd.DataFrame, short_labels: bool = True) -> pd.DataFrame:
    if not short_labels: return df

    def shorten_group_name(group_name: str) -> str:
        if '=' in group_name:
            parts = []
            for param in group_name.split(',')[1:]:
                if '=' in param:
                    key, value = param.split('=', 1)
                    key = '_'.join(key.split('_')[1:]) if key.startswith('redo_') else key
                    initials = ''.join([w[0] for w in key.split('_')]) if '_' in key else key[:3]
                    parts.append(f"{initials}={value}")
                else:
                    parts.append(param)
            return ','.join(parts)
        elif '_' in group_name:
            parts = group_name.split('_')
            return '_'.join([p if i == 0 or (p.startswith('s') and p[1:].isdigit()) else p[:3]
                           for i, p in enumerate(parts)])
        return group_name

    df_copy = df.copy()
    df_copy['short_group'] = df_copy['group'].apply(shorten_group_name)
    return df_copy


def aggregate_data(df: pd.DataFrame, grouping_mode: Literal["parameter", "seed", "config"] = "parameter") -> pd.DataFrame:
    if grouping_mode == "seed":
        param_cols = [col for col in df.columns if col not in ['algorithm', 'seed', 'step', 'value', 'run_name', 'group']]
        config_str = df[param_cols].apply(lambda row: ','.join([f"{k}={v}" for k, v in row.items()]), axis=1)
        df['group'] = df['algorithm'] + '_' + config_str + '_' + df['seed']
    elif grouping_mode == "config":
        param_cols = [col for col in df.columns if col not in ['algorithm', 'seed', 'step', 'value', 'run_name', 'group']]
        config_str = df[param_cols].apply(lambda row: ','.join([f"{k}={v}" for k, v in row.items()]), axis=1)
        df['group'] = df['algorithm'] + '_' + config_str

    df['binned_step'] = (df['step'] / 10_000).round() * 10_000

    aggregated_data = []
    for (group_name, step), group_df in df.groupby(['group', 'binned_step']):
        values = group_df['value'].values
        if len(values) > 0:
            iqm_value = compute_iqm(values)
            q25, q75 = np.percentile(values, [25, 75]) if len(values) > 1 else (iqm_value, iqm_value)
            aggregated_data.append({
                'group': group_name, 'step': step / 1_000_000, 'iqm': iqm_value,
                'q25': q25, 'q75': q75, 'n_runs': len(values)
            })

    return pd.DataFrame(aggregated_data).sort_values(['group', 'step'])


def create_ablation_chart(df: pd.DataFrame, metric: str, title: str = "", use_short_labels: bool = True) -> alt.Chart:
    chart_df = create_short_labels(df, use_short_labels)
    group_col = 'short_group' if use_short_labels and 'short_group' in chart_df.columns else 'group'

    base = alt.Chart(chart_df).encode(
        x=alt.X('step:Q', title='Training Steps (Millions)',
               scale=alt.Scale(domain=[0, chart_df['step'].max() * 1.1])),
        color=alt.Color(f'{group_col}:N', title='Configuration')
    )

    smoothed_base = base.transform_window(
        frame=[-5, 5], groupby=[group_col],
        smooth_iqm='mean(iqm)', smooth_q25='mean(q25)', smooth_q75='mean(q75)'
    )

    bands = smoothed_base.mark_area(opacity=0.2).encode(
        y=alt.Y('smooth_q25:Q', title=metric.replace('_', ' ').title()),
        y2=alt.Y2('smooth_q75:Q')
    )

    lines = smoothed_base.mark_line(strokeWidth=2).encode(
        y=alt.Y('smooth_iqm:Q', title=""),
        tooltip=[alt.Tooltip(f'{group_col}:N', title='Group'),
                alt.Tooltip('step:Q', title='Step (M)', format='.2f'),
                alt.Tooltip('smooth_iqm:Q', title='IQM', format='.3f')]
    )

    return (bands + lines).properties(
        width=800, height=400,
        title=title or f"{metric.replace('_', ' ').title()} Ablation Study"
    ).configure_axis(grid=True, gridOpacity=0.3).interactive()


def main(wandb_entity: str, wandb_project: str = "crl_experiments", group: str = "slippery_ant_ccbp_sweep",
         metric: str = "charts/mean_episodic_return", split_by: Optional[str] = None,
         grouping_mode: Literal["parameter", "seed", "config"] = "parameter", top_k: Optional[int] = None,
         ranking_criteria: Literal["final", "peak", "average", "auc"] = "final", short_labels: bool = True,
         output_dir: str = "./plots/ablation", ext: str = "svg", debug: bool = False):
    df = fetch_ablation_data(wandb_entity, wandb_project, group, metric, split_by)
    if df.empty: return print(f"No data found for group: '{group}'")

    agg_df = aggregate_data(df, grouping_mode)
    if agg_df.empty: return print("No aggregated data available")
    if top_k and top_k > 0:
        ranking = {'final': agg_df.groupby('group')['iqm'].last(),
                  'peak': agg_df.groupby('group')['iqm'].max(),
                  'average': agg_df.groupby('group')['iqm'].mean(),
                  'auc': pd.Series({g: np.trapz(gdf['iqm'], gdf['step'])
                                   for g, gdf in agg_df.groupby('group') if len(gdf) > 1})}
        top_groups = ranking[ranking_criteria].sort_values(ascending=False).head(top_k).index
        agg_df = agg_df[agg_df['group'].isin(top_groups)]
        if agg_df.empty: return
    title_parts = [group.replace('_', ' ').title()]
    if split_by: title_parts.append(f"by {split_by}")
    if top_k: title_parts.append(f"(top {top_k} by {ranking_criteria})")
    title_parts.append(f"({grouping_mode} mode)")
    title = ": ".join(title_parts)

    final_data = agg_df.groupby('group')['iqm'].last().sort_values(ascending=False)
    print(f"\nTop performers: {dict(final_data.head(3))}")

    chart = create_ablation_chart(agg_df, metric, title, short_labels)

    save_name = f"{group}_{metric.replace('/', '_')}"
    if split_by: save_name += f"_by_{split_by}"
    if top_k: save_name += f"_top{top_k}_{ranking_criteria}"
    save_name += f"_{grouping_mode}_ablation.{ext}"

    save_path = Path(output_dir) / save_name
    save_path.parent.mkdir(exist_ok=True, parents=True)
    chart.save(str(save_path))
    print(f"Chart saved to: {save_path}")
    return chart


if __name__ == "__main__":
    tyro.cli(main)
