#!/usr/bin/env python3
import tyro
import numpy as np
import altair as alt
from pathlib import Path
import wandb
import pandas as pd
from collections import defaultdict
from typing import Optional, Literal
import re


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


def parse_run_name(run_name: str) -> dict:
    """Parse run name to extract algorithm and hyperparameters.
    
    Expected format: algo_param1=value1,param2=value2_sXX
    """
    parts = run_name.split('_')
    if len(parts) < 3:
        return {"algorithm": run_name, "seed": "0"}
    
    # Extract seed (last part starting with 's')
    seed = parts[-1] if parts[-1].startswith('s') else "0"
    
    # Extract algorithm (first part)
    algorithm = parts[0]
    
    # Join middle parts and parse as comma-separated key=value pairs
    tag_part = '_'.join(parts[1:-1])
    params = {}
    
    # Split by comma and parse each key=value pair
    for param_pair in tag_part.split(','):
        if '=' in param_pair:
            key, value = param_pair.split('=', 1)
            try:
                # Try to convert to float first, then int, otherwise keep as string
                if '.' in value or 'e' in value.lower():
                    params[key] = float(value)
                else:
                    try:
                        params[key] = int(value)
                    except ValueError:
                        params[key] = float(value)
            except ValueError:
                params[key] = value
    
    return {
        "algorithm": algorithm,
        "seed": seed,
        **params
    }


def fetch_runs(entity: str, project: str, group: str):
    api = wandb.Api()
    filters = {"group": group}
    runs = api.runs(f"{entity}/{project}", filters=filters, per_page=300)
    all_runs = list(runs)
    
    print(f"Total runs fetched: {len(all_runs)}")
    
    run_states = defaultdict(int)
    for run in all_runs:
        run_states[run.state] += 1
    print(f"Run states: {dict(run_states)}")
    
    return [x for x in all_runs if x.state == "finished"]


def fetch_ablation_data(
    entity: str, 
    project: str, 
    group: str, 
    metric: str,
    split_by: Optional[str] = None
) -> pd.DataFrame:
    """Fetch data and organize by hyperparameter splits."""
    runs = fetch_runs(entity, project, group)
    
    # Parse all runs and extract hyperparameters
    run_data = []
    parsed_examples = {}
    
    for run in runs:
        parsed = parse_run_name(run.name)
        
        # Store a few examples for debugging
        if len(parsed_examples) < 3:
            parsed_examples[run.name] = parsed
        
        # Get metric history
        history = run.history(keys=[metric, "_step"])
        if history.empty:
            print(f"Warning: Run {run.name} has no data for metric '{metric}'")
            continue
        
        for _, row in history.iterrows():
            step = row.get("_step")
            value = row.get(metric)
            if pd.notna(step) and pd.notna(value):
                run_data.append({
                    **parsed,
                    "step": step,
                    "value": value,
                    "run_name": run.name
                })
    
    # Show parsing examples
    print("Run name parsing examples:")
    for run_name, parsed in parsed_examples.items():
        print(f"  {run_name} -> {parsed}")
    
    if not run_data:
        return pd.DataFrame()
    
    df = pd.DataFrame(run_data)
    
    # Create grouping column based on split_by parameter
    if split_by:
        if split_by in df.columns:
            print("split_by:\n", split_by)
            df['group'] = f"{split_by}=" + df[split_by].astype(str)
        else:
            print(f"Warning: Parameter '{split_by}' not found in data. Available parameters:")
            param_cols = [col for col in df.columns if col not in ['algorithm', 'seed', 'step', 'value', 'run_name']]
            print(f"  {param_cols}")
            df['group'] = df['algorithm']
    else:
        df['group'] = df['algorithm']
    
    return df


def create_short_labels(df: pd.DataFrame, short_labels: bool = True) -> pd.DataFrame:
    """Create shortened labels for legend display."""
    if not short_labels:
        return df
    
    def shorten_group_name(group_name: str) -> str:
        # Handle different group name formats
        
        # Case 1: Simple algorithm names (no shortening needed)
        if not ('=' in group_name or '_' in group_name or ',' in group_name):
            return group_name
        
        # Case 2: Names with = signs (parameter=value format like "redo_tx_lr=0.001,update_frequency=1000.0,score_threshold=0.001")
        if '=' in group_name:
            # Split by comma first to handle comma-separated params
            param_parts = group_name.split(',')
            shortened_parts = []
            
            for param in param_parts[1:]:
                if '=' in param:
                    key, value = param.split('=', 1)
                    
                    # Remove algorithm prefix if present (e.g., "redo_tx_lr" -> "tx_lr")
                    if '_' in key and key.split('_')[0] == 'redo':
                        key = '_'.join(key.split('_')[1:])
                    
                    # Create initials from key
                    if '_' in key:
                        initials = ''.join([word[0] for word in key.split('_')])
                    else:
                        initials = key[:3] if len(key) > 3 else key
                    
                    shortened_parts.append(f"{initials}={value}")
                else:
                    shortened_parts.append(param)
            
            return ','.join(shortened_parts)
        
        # Case 3: Names with underscores but no = (algorithm_param1_param2_seed format)
        if '_' in group_name:
            parts = group_name.split('_')
            shortened_parts = []
            
            for i, part in enumerate(parts):
                # Keep algorithm name (first part) and seed (if it starts with 's')
                if i == 0 or (part.startswith('s') and part[1:].isdigit()):
                    shortened_parts.append(part)
                else:
                    # Shorten parameter names
                    if len(part) > 3:
                        shortened_parts.append(part[:3])
                    else:
                        shortened_parts.append(part)
            
            return '_'.join(shortened_parts)
        
        # Default: return unchanged
        return group_name
    
    df_copy = df.copy()
    df_copy['short_group'] = df_copy['group'].apply(shorten_group_name)
    return df_copy


def aggregate_data(df: pd.DataFrame, grouping_mode: Literal["parameter", "seed", "config"] = "parameter") -> pd.DataFrame:
    """Aggregate data based on different grouping strategies."""
    
    if grouping_mode == "parameter":
        # Group by parameter value, ignoring other parameters and averaging across seeds
        group_cols = ['group', 'step']
        
    elif grouping_mode == "seed":
        # Group by specific configuration and seed (no averaging across seeds)
        param_cols = [col for col in df.columns if col not in ['algorithm', 'seed', 'step', 'value', 'run_name', 'group']]
        config_str = df[param_cols].apply(
            lambda row: ','.join([f"{k}={v}" for k, v in row.items()]), 
            axis=1
        )
        df['config'] = df['algorithm'] + '_' + config_str
        df['group'] = df['config'] + '_' + df['seed']
        group_cols = ['group', 'step']
        
    else:  # config
        # Group by specific configuration, averaging across seeds
        param_cols = [col for col in df.columns if col not in ['algorithm', 'seed', 'step', 'value', 'run_name', 'group']]
        config_str = df[param_cols].apply(
            lambda row: ','.join([f"{k}={v}" for k, v in row.items()]), 
            axis=1
        )
        df['group'] = df['algorithm'] + '_' + config_str
        group_cols = ['group', 'step']
    
    # Bin steps and aggregate
    df['binned_step'] = (df['step'] / 10_000).round() * 10_000
    
    aggregated_data = []
    for group_vals, group_df in df.groupby(['group', 'binned_step']):
        group_name, step = group_vals
        values = group_df['value'].values
        
        if len(values) > 0:
            iqm_value = compute_iqm(values)
            q25, q75 = np.percentile(values, [25, 75]) if len(values) > 1 else (iqm_value, iqm_value)
            
            aggregated_data.append({
                'group': group_name,
                'step': step / 1_000_000,  # Convert to millions
                'iqm': iqm_value,
                'q25': q25,
                'q75': q75,
                'n_runs': len(values)
            })
    
    result_df = pd.DataFrame(aggregated_data).sort_values(['group', 'step'])
    
    print(f"\nData summary ({grouping_mode} mode):")
    for group in result_df['group'].unique():
        group_df = result_df[result_df['group'] == group]
        max_runs = group_df['n_runs'].max()
        total_points = len(group_df)
        print(f"  - {group}: up to {max_runs} runs, {total_points} aggregated data points")
    
    return result_df


def create_ablation_chart(df: pd.DataFrame, metric: str, title: str = "", use_short_labels: bool = True) -> alt.Chart:
    """Create Altair chart for ablation study."""
    
    # Create shortened labels if requested
    chart_df = create_short_labels(df, use_short_labels)
    group_col = 'short_group' if use_short_labels and 'short_group' in chart_df.columns else 'group'
    
    base = alt.Chart(chart_df).encode(
        x=alt.X('step:Q', 
                title='Training Steps (Millions)',
                scale=alt.Scale(domain=[0, chart_df['step'].max() * 1.1], nice=True),
                axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
        color=alt.Color(f'{group_col}:N', 
                        title='Configuration',
                        legend=alt.Legend(
                            title=None,
                            symbolOpacity=1.0,
                            orient='right',
                            fillColor='rgba(255,255,255,0.8)',
                            strokeColor='gray',
                            padding=5,
                            cornerRadius=3,
                            labelFontSize=10,
                            symbolSize=100
                        ))
    )

    # Apply smoothing
    smoothed_base = base.transform_window(
        frame=[-5, 5],
        groupby=[group_col],
        smooth_iqm='mean(iqm)',
        smooth_q25='mean(q25)',
        smooth_q75='mean(q75)'
    )

    # Shaded area for IQR
    bands = smoothed_base.mark_area(opacity=0.2).encode(
        y=alt.Y('smooth_q25:Q', title=metric.replace('_', ' ').title(),
                axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
        y2=alt.Y2('smooth_q75:Q', title=""),
    )

    # Line for IQM
    lines = smoothed_base.mark_line(strokeWidth=2).encode(
        y=alt.Y('smooth_iqm:Q', title=""),
        tooltip=[
            alt.Tooltip(f'{group_col}:N', title='Group'),
            alt.Tooltip('step:Q', title='Step (M)', format='.2f'),
            alt.Tooltip('smooth_iqm:Q', title='Smoothed IQM', format='.3f'),
            alt.Tooltip('n_runs:Q', title='Runs')
        ]
    )
    
    chart = bands + lines
    
    return chart.properties(
        width=800,
        height=400,
        title=alt.TitleParams(
            text=title if title else f"{metric.replace('_', ' ').title()} Ablation Study",
            fontSize=16,
            fontWeight='bold'
        )
    ).configure_axis(
        grid=True,
        gridOpacity=0.3
    ).interactive()


def main(
    wandb_entity: str,
    wandb_project: str = "crl_experiments",
    group: str = "slippery_ant_ccbp_sweep",
    metric: str = "charts/mean_episodic_return",
    split_by: Optional[str] = None,  # e.g., "replacement_rate", "maturity_threshold"
    grouping_mode: Literal["parameter", "seed", "config"] = "parameter",
    top_k: Optional[int] = None,  # Show only top K configurations
    ranking_criteria: Literal["final", "peak", "average", "auc"] = "final",
    short_labels: bool = True,  # Use shortened labels in legend
    output_dir: str = "./plots/ablation",
    ext: str = "svg",
    debug: bool = False
):
    """
    Create ablation plots for hyperparameter sweeps.
    
    Args:
        wandb_entity: W&B entity name
        wandb_project: W&B project name  
        group: W&B group name for the sweep
        metric: Metric to plot (e.g., "charts/mean_episodic_return")
        split_by: Hyperparameter to split by (e.g., "replacement_rate")
        grouping_mode: How to group data:
            - "parameter": Group by split_by parameter value, average across seeds
            - "seed": Group by full config + seed (no averaging)
            - "config": Group by full config, average across seeds
        top_k: Show only top K configurations (None = show all)
        ranking_criteria: Criteria for selecting top-k:
            - "final": Final performance (last step)
            - "peak": Peak performance (best achieved)
            - "average": Average performance across training
            - "auc": Area under curve (cumulative performance)
        short_labels: Use shortened parameter names in legend (e.g., "rtr=0.001" instead of "redo_tx_rl=0.001")
        output_dir: Directory to save plots
        ext: File extension for saved plots
        debug: Enable debug mode
    """
    
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    print(f"Fetching data for group: '{group}', metric: '{metric}'")
    if split_by:
        print(f"Splitting by parameter: '{split_by}'")
    print(f"Grouping mode: '{grouping_mode}'")
    if top_k:
        print(f"Showing top {top_k} configurations by {ranking_criteria} performance")
    
    df = fetch_ablation_data(wandb_entity, wandb_project, group, metric, split_by)
    
    if df.empty:
        print(f"No data found for group: '{group}', metric: '{metric}'")
        return
    
    # Print available parameters for reference
    param_cols = [col for col in df.columns if col not in ['algorithm', 'seed', 'step', 'value', 'run_name', 'group']]
    print(f"Available parameters: {param_cols}")
    
    # Aggregate data
    agg_df = aggregate_data(df, grouping_mode)
    
    if agg_df.empty:
        print("No aggregated data available")
        return
    
    # Apply top-k filtering if requested
    if top_k is not None and top_k > 0:
        # Calculate ranking metrics
        final_step_data = agg_df.groupby('group')['iqm'].last().sort_values(ascending=False)
        peak_data = agg_df.groupby('group')['iqm'].max().sort_values(ascending=False)
        mean_data = agg_df.groupby('group')['iqm'].mean().sort_values(ascending=False)
        
        # Calculate AUC for ranking
        auc_data = {}
        for group in agg_df['group'].unique():
            group_df = agg_df[agg_df['group'] == group].sort_values('step')
            if len(group_df) > 1:
                auc = np.trapz(group_df['iqm'], group_df['step'])
                auc_data[group] = auc
        auc_series = pd.Series(auc_data).sort_values(ascending=False)
        
        # Select top-k groups based on criteria
        if ranking_criteria == "final":
            top_groups = final_step_data.head(top_k).index.tolist()
        elif ranking_criteria == "peak":
            top_groups = peak_data.head(top_k).index.tolist()
        elif ranking_criteria == "average":
            top_groups = mean_data.head(top_k).index.tolist()
        elif ranking_criteria == "auc":
            top_groups = auc_series.head(top_k).index.tolist()
        else:
            top_groups = final_step_data.head(top_k).index.tolist()
        
        # Filter aggregated data to only include top groups
        agg_df = agg_df[agg_df['group'].isin(top_groups)]
        print(f"Filtered to top {len(top_groups)} configurations: {top_groups}")
        
        if agg_df.empty:
            print("No data remaining after filtering")
            return
    
    # Create title
    title_parts = [group.replace('_', ' ').title()]
    if split_by:
        title_parts.append(f"by {split_by}")
    if top_k:
        title_parts.append(f"(top {top_k} by {ranking_criteria})")
    title_parts.append(f"({grouping_mode} mode)")
    title = ": ".join(title_parts)
    
    # Print comprehensive performance statistics
    final_step_data = agg_df.groupby('group')['iqm'].last().sort_values(ascending=False)
    peak_data = agg_df.groupby('group')['iqm'].max().sort_values(ascending=False)
    mean_data = agg_df.groupby('group')['iqm'].mean().sort_values(ascending=False)
    std_data = agg_df.groupby('group')['iqm'].std()
    
    # Calculate area under curve (AUC) for each group
    auc_data = {}
    for group in agg_df['group'].unique():
        group_df = agg_df[agg_df['group'] == group].sort_values('step')
        if len(group_df) > 1:
            # Simple trapezoidal rule for AUC
            auc = np.trapz(group_df['iqm'], group_df['step'])
            auc_data[group] = auc
    auc_series = pd.Series(auc_data).sort_values(ascending=False)
    
    # Calculate learning efficiency (time to reach certain thresholds)
    efficiency_data = {}
    if len(peak_data) > 0:
        # Use 50%, 75%, 90% of best peak as thresholds
        best_peak = peak_data.iloc[0]
        thresholds = [0.5 * best_peak, 0.75 * best_peak, 0.9 * best_peak]
        
        for threshold_pct, threshold in zip([50, 75, 90], thresholds):
            threshold_steps = {}
            for group in agg_df['group'].unique():
                group_df = agg_df[agg_df['group'] == group].sort_values('step')
                reached = group_df[group_df['iqm'] >= threshold]
                if not reached.empty:
                    threshold_steps[group] = reached.iloc[0]['step']
            if threshold_steps:
                efficiency_data[threshold_pct] = pd.Series(threshold_steps).sort_values()
    
    if len(final_step_data) > 0:
        print(f"\n{'='*80}")
        print(f"PERFORMANCE SUMMARY")
        print(f"{'='*80}")
        
        # Final performance ranking
        print(f"\nFINAL PERFORMANCE (at last step):")
        print(f"{'Rank':<4} {'Configuration':<40} {'Value':<12} {'Std Dev':<12}")
        print(f"{'-'*70}")
        for i, (config, value) in enumerate(final_step_data.items(), 1):
            std_val = std_data.get(config, 0)
            print(f"{i:<4} {config:<40} {value:<12.4f} ±{std_val:<11.4f}")
        
        # Peak performance ranking  
        print(f"\nPEAK PERFORMANCE (best achieved):")
        print(f"{'Rank':<4} {'Configuration':<40} {'Value':<12}")
        print(f"{'-'*58}")
        for i, (config, value) in enumerate(peak_data.items(), 1):
            print(f"{i:<4} {config:<40} {value:<12.4f}")
            
        # Mean performance ranking
        print(f"\nAVERAGE PERFORMANCE (mean across training):")
        print(f"{'Rank':<4} {'Configuration':<40} {'Value':<12} {'Std Dev':<12}")
        print(f"{'-'*70}")
        for i, (config, value) in enumerate(mean_data.items(), 1):
            std_val = std_data.get(config, 0)
            print(f"{i:<4} {config:<40} {value:<12.4f} ±{std_val:<11.4f}")
        
        # Area under curve ranking
        if auc_series.size > 0:
            print(f"\nCUMULATIVE PERFORMANCE (area under curve):")
            print(f"{'Rank':<4} {'Configuration':<40} {'AUC':<12}")
            print(f"{'-'*58}")
            for i, (config, auc) in enumerate(auc_series.items(), 1):
                print(f"{i:<4} {config:<40} {auc:<12.2f}")
        
        # Learning efficiency
        if efficiency_data:
            print(f"\nLEARNING EFFICIENCY (steps to reach performance thresholds):")
            for threshold_pct, threshold_series in efficiency_data.items():
                if not threshold_series.empty:
                    threshold_val = thresholds[[50, 75, 90].index(threshold_pct)]
                    print(f"\n   {threshold_pct}% of best peak ({threshold_val:.4f}):")
                    print(f"   {'Rank':<4} {'Configuration':<40} {'Steps (M)':<12}")
                    print(f"   {'-'*58}")
                    for i, (config, steps) in enumerate(threshold_series.items(), 1):
                        print(f"   {i:<4} {config:<40} {steps:<12.2f}")
        
        # Summary statistics
        final_values = final_step_data.values
        peak_values = peak_data.values
        print(f"\n📋 SUMMARY STATISTICS:")
        print(f"   Number of configurations: {len(final_step_data)}")
        print(f"   Final performance range: {final_values.min():.4f} - {final_values.max():.4f}")
        print(f"   Final performance spread: {final_values.max() - final_values.min():.4f}")
        print(f"   Peak performance range: {peak_values.min():.4f} - {peak_values.max():.4f}")
        print(f"   Peak performance spread: {peak_values.max() - peak_values.min():.4f}")
        print(f"   Final vs Peak gap (best): {peak_data.iloc[0] - final_step_data.loc[peak_data.index[0]]:.4f}")
        
        print(f"\n{'='*80}")
    
    # Create chart
    chart = create_ablation_chart(agg_df, metric, title, short_labels)
    
    # Save chart
    save_name = f"{group}_{metric.replace('/', '_')}"
    if split_by:
        save_name += f"_by_{split_by}"
    if top_k:
        save_name += f"_top{top_k}_{ranking_criteria}"
    save_name += f"_{grouping_mode}_ablation.{ext}"
    
    save_path = Path(output_dir) / save_name
    save_path.parent.mkdir(exist_ok=True, parents=True)
    chart.save(str(save_path))
    print(f"\nChart saved to: {save_path}")
    
    return chart


if __name__ == "__main__":
    tyro.cli(main)
