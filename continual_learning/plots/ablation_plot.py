#!/usr/bin/env python3
import tyro
import numpy as np
import altair as alt
from pathlib import Path
import wandb
import pandas as pd
import re
from collections import defaultdict
from typing import Optional, Literal, List, Dict, Iterable

ALT_THEME_NAME = "times_new_roman_theme"
ALT_FONT_FAMILY = "Times New Roman"


def scaled_font_size(base_size: float, factor: float) -> int:
    size = int(round(base_size * factor))
    return max(size, 1)


def _times_new_roman_theme() -> dict:
    return {
        "config": {
            "title": {"font": ALT_FONT_FAMILY},
            "axis": {
                "labelFont": ALT_FONT_FAMILY,
                "titleFont": ALT_FONT_FAMILY,
            },
            "header": {
                "labelFont": ALT_FONT_FAMILY,
                "titleFont": ALT_FONT_FAMILY,
            },
            "legend": {
                "labelFont": ALT_FONT_FAMILY,
                "titleFont": ALT_FONT_FAMILY,
            },
            "mark": {"font": ALT_FONT_FAMILY},
            "text": {"font": ALT_FONT_FAMILY},
        }
    }


def _enable_times_new_roman_theme() -> None:
    try:
        alt.themes.register(ALT_THEME_NAME, _times_new_roman_theme)
    except ValueError:
        pass
    alt.themes.enable(ALT_THEME_NAME)


_enable_times_new_roman_theme()

RENAME_LEGEND = {"exp": "Exponential",
          "sigmoid": "Sigmoid",
          "linear": "Linear",
          "softplus": "Softplus"}

PARAMETER_SYMBOLS = {
    "replacement_rate": "ρ",
    "sharpness": "κ",
}

ALGORITHM_LEGEND_ORDER = [
    "CCBP",
    "CCBP",
    "CBP",
    "ReGraMa",
    "ReDo",
    "Shrink & Perturb",
    "Adam",
]

_ALGORITHM_ORDER_LOOKUP = {}
for _idx, _name in enumerate(ALGORITHM_LEGEND_ORDER):
    _ALGORITHM_ORDER_LOOKUP.setdefault(_name, _idx)


def _algorithm_base_label(label: str) -> str:
    return label.split(',', 1)[0].strip() if label else ""


def _algorithm_legend_sort_key(label: str) -> tuple[int, str]:
    base = _algorithm_base_label(label)
    order_index = _ALGORITHM_ORDER_LOOKUP.get(base, len(ALGORITHM_LEGEND_ORDER))
    return order_index, label


def _extract_numeric_from_label(label: str) -> Optional[float]:
    """Extract numeric value from a parameter label like 'ρ=0.1' or '0.1'."""
    if '=' in label:
        # Extract value after '='
        value_str = label.split('=', 1)[1].split(',')[0].strip()
    else:
        value_str = label.strip()

    try:
        # Try to parse as float
        return float(value_str)
    except ValueError:
        return None


def _parameter_value_sort_key(label: str) -> tuple[float, str]:
    """Sort by numeric value, then by label for ties."""
    numeric_value = _extract_numeric_from_label(label)
    if numeric_value is not None:
        return (numeric_value, label)
    # If no numeric value found, put at the end
    return (float('inf'), label)


def build_algorithm_legend_domain(labels: Iterable[str], sort_by_value: bool = False) -> List[str]:
    cleaned = {
        label.strip(): None
        for label in labels
        if isinstance(label, str) and label.strip()
    }
    if sort_by_value:
        return sorted(cleaned.keys(), key=_parameter_value_sort_key)
    return sorted(cleaned.keys(), key=_algorithm_legend_sort_key)


def compute_iqm(values: np.ndarray) -> float:
    if len(values) < 4: return np.mean(values)
    q25, q75 = np.percentile(values, [25, 75])
    mask = (values >= q25) & (values <= q75)
    return np.mean(values[mask]) if np.any(mask) else np.mean(values)


def coerce_numeric_value(value) -> Optional[float]:
    """Best-effort conversion of W&B history values to finite floats."""

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) if np.isfinite(value) else None

    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        lowered = cleaned.lower()
        if lowered in {"nan", "inf", "+inf", "-inf"}:
            return None
        try:
            numeric = float(cleaned)
        except ValueError:
            matches = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
            if len(matches) != 1:
                return None
            try:
                numeric = float(matches[0])
            except ValueError:
                return None
        return float(numeric) if np.isfinite(numeric) else None

    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            arr = np.asarray(value, dtype=np.float64).reshape(-1)
        except (ValueError, TypeError):
            return None
        finite = arr[np.isfinite(arr)]
        if finite.size == 1:
            return float(finite[0])
        return None

    return None


def parse_run_name(run_name: str) -> dict:
    if '_' not in run_name:
        return {"algorithm": run_name, "seed": "0"}

    algorithm, remainder = run_name.split('_', 1)
    params = {}
    seed_value: Optional[str] = None

    for param_pair in remainder.split(','):
        if '=' not in param_pair:
            continue

        key, value = param_pair.split('=', 1)
        try:
            converted = float(value) if '.' in value or 'e' in value.lower() else int(value)
        except ValueError:
            converted = value

        params[key] = converted
        if key == 'seed':
            seed_value = str(converted)

    if 'seed' in params:
        params.pop('seed')

    if seed_value is None:
        match = re.search(r'_s(\d+)$', run_name)
        seed_value = match.group(1) if match else "0"

    return {"algorithm": algorithm, "seed": seed_value, **params}


def slugify_title(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    sanitized = sanitized.strip("_")
    return sanitized.lower() or "plot"


def fetch_runs(entity: str, project: str, group: str, created_after: Optional[str] = None, include_failed: bool = False):
    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters={"group": group}, per_page=300))

    # Filter by state
    if include_failed:
        valid_runs = [r for r in runs if r.state in ["finished", "failed"]]
    else:
        valid_runs = [r for r in runs if r.state == "finished"]

    # Filter by creation date if specified
    if created_after:
        from datetime import datetime, timezone
        # Parse cutoff date - add timezone if not present
        if 'T' in created_after:
            cutoff_date = datetime.fromisoformat(created_after.replace('Z', '+00:00'))
        else:
            # Assume midnight UTC for date-only strings
            cutoff_date = datetime.fromisoformat(created_after + "T00:00:00+00:00")

        filtered_runs = []
        for run in valid_runs:
            # Parse run date and ensure it has timezone info
            run_date_str = run.created_at
            if isinstance(run_date_str, str):
                run_date = datetime.fromisoformat(run_date_str.replace('Z', '+00:00'))
            else:
                # If it's already a datetime, ensure it has timezone
                run_date = run_date_str
                if run_date.tzinfo is None:
                    run_date = run_date.replace(tzinfo=timezone.utc)

            # Ensure cutoff_date has timezone
            if cutoff_date.tzinfo is None:
                cutoff_date = cutoff_date.replace(tzinfo=timezone.utc)

            if run_date >= cutoff_date:
                filtered_runs.append(run)

        status_msg = "finished/failed" if include_failed else "finished"
        print(f"Fetched {len(runs)} runs, {len(valid_runs)} {status_msg}, {len(filtered_runs)} after {created_after}")
        return filtered_runs

    status_msg = "finished/failed" if include_failed else "finished"
    print(f"Fetched {len(runs)} runs, {len(valid_runs)} {status_msg}")
    return valid_runs


def format_hyperparameter_value(value, force_scientific: bool = False) -> str:
    """Format hyperparameter values consistently for display and grouping.

    Args:
        value: The hyperparameter value to format
        force_scientific: If True, use scientific notation for all numeric values
    """
    if isinstance(value, (int, np.integer)):
        if force_scientific and value != 0:
            return f"{float(value):.0e}"
        return str(int(value))
    elif isinstance(value, (float, np.floating)):
        # Handle zero specially to avoid '0e+00' formatting
        if value == 0.0:
            return "0"
        if force_scientific:
            # Use scientific notation for all values
            return f"{value:.0e}"
        # Use scientific notation for very small or large numbers
        if abs(value) < 0.0001 or abs(value) >= 10000:
            return f"{value:.0e}"  # e.g., "1e-06", "1e+04"
        else:
            # Standard notation, removing trailing zeros
            formatted = f"{value:.6f}".rstrip('0').rstrip('.')
            return formatted
    else:
        return str(value)


def fetch_ablation_data(entity: str, project: str, group: str, metric: str, split_by: Optional[str] = None, show_metric_in_legend: bool = False, created_after: Optional[str] = None, force_scientific: bool = False, include_failed: bool = False) -> pd.DataFrame:
    runs = fetch_runs(entity, project, group, created_after, include_failed)
    run_data = []

    for run in runs:
        parsed = parse_run_name(run.name)
        history = run.history(keys=[metric, "_step"], samples=5000)
        if history.empty: continue

        for _, row in history.iterrows():
            step, value = row.get("_step"), row.get(metric)
            numeric_value = coerce_numeric_value(value)
            if pd.notna(step) and numeric_value is not None:
                run_data.append({**parsed, "step": step, "value": numeric_value, "run_name": run.name})

    if not run_data: return pd.DataFrame()

    df = pd.DataFrame(run_data)
    if split_by and split_by in df.columns:
        def normalized_split_value(row):
            value = row[split_by]
            if isinstance(value, str):
                match = re.match(r'(.+)_s(\d+)$', value)
                if match and str(row.get('seed')) == match.group(2):
                    return match.group(1)
            return value

        normalized_values = df.apply(normalized_split_value, axis=1)

        # Format values consistently for proper grouping and display
        formatted_values = normalized_values.apply(lambda v: format_hyperparameter_value(v, force_scientific=force_scientific))
        # Use symbol notation if available (e.g., ρ for replacement_rate)
        param_name = PARAMETER_SYMBOLS.get(split_by, split_by)
        df['group'] = formatted_values.apply(lambda value: f"{param_name}={value}" if show_metric_in_legend else value)
    else:
        df['group'] = df['algorithm']

    return df


def create_short_labels(df: pd.DataFrame, short_labels: bool = True) -> pd.DataFrame:
    if not short_labels: return df

    def shorten_group_name(group_name: str) -> str:
        if '=' in group_name:
            segments = group_name.split(',')
            if len(segments) == 1:
                return group_name

            parts = []
            for param in segments[1:]:
                if '=' in param:
                    key, value = param.split('=', 1)
                    key = '_'.join(key.split('_')[1:]) if key.startswith('redo_') else key
                    initials = ''.join([w[0] for w in key.split('_')]) if '_' in key else key[:3]
                    parts.append(f"{initials}={value}")
                else:
                    parts.append(param)
            return ','.join(parts) if parts else segments[0]
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
        config_str = df[param_cols].apply(lambda row: ','.join([f"{v}" for k, v in row.items()]), axis=1)
        df['group'] = df['algorithm'] + '_' + config_str + '_' + df['seed']
    elif grouping_mode == "config":
        param_cols = [col for col in df.columns if col not in ['algorithm', 'seed', 'step', 'value', 'run_name', 'group']]
        config_str = df[param_cols].apply(lambda row: ','.join([f"{k}={v}" for k, v in row.items()]), axis=1)
        df['group'] = df['algorithm'] + '_' + config_str

    df['binned_step'] = (df['step'] / 10_000).round() * 10_000

    aggregated_data = []
    for (group_name, step), group_df in df.groupby(['group', 'binned_step']):
        values = pd.to_numeric(group_df['value'], errors='coerce').dropna().to_numpy(dtype=np.float64)
        if len(values) > 0:
            iqm_value = compute_iqm(values)
            q25, q75 = np.percentile(values, [25, 75]) if len(values) > 1 else (iqm_value, iqm_value)
            aggregated_data.append({
                'group': group_name, 'step': step / 1_000_000, 'iqm': iqm_value,
                'q25': q25, 'q75': q75, 'n_runs': len(values)
            })
    result_df = pd.DataFrame(aggregated_data)
    if result_df.empty:
        return result_df

    result_df['group'] = result_df['group'].apply(lambda name: RENAME_LEGEND.get(name, name))
    return result_df.sort_values(['group', 'step'])


def _format_interval(value: float, lower_quantile: Optional[float], upper_quantile: Optional[float]) -> str:
    if value is None or not np.isfinite(value):
        return "nan (+0.000/-0.000)"
    lower = max(value - lower_quantile, 0.0) if lower_quantile is not None and pd.notna(lower_quantile) else 0.0
    upper = max(upper_quantile - value, 0.0) if upper_quantile is not None and pd.notna(upper_quantile) else 0.0
    return f"{value:.3f} (+{upper:.3f}/-{lower:.3f})"


def summarize_group_performance(df: pd.DataFrame, metric: str) -> List[Dict[str, float]]:
    if df.empty:
        print("\nNo aggregated performance data to summarise.")
        return []

    grouped = list(df.groupby('group', sort=True))
    print("\nData summary:")
    for group_name, group_df in grouped:
        max_runs = None
        if 'n_runs' in group_df:
            run_counts = pd.to_numeric(group_df['n_runs'], errors='coerce').dropna()
            if not run_counts.empty:
                max_runs = int(run_counts.max())
        if max_runs is not None:
            print(f"  - {group_name}: up to {max_runs} runs, {len(group_df)} aggregated data points")
        else:
            print(f"  - {group_name}: {len(group_df)} aggregated data points")

    print(f"    Performance (IQM) for {metric}:")
    summaries: List[Dict[str, float]] = []
    for group_name, group_df in grouped:
        group_df = group_df.sort_values('step')
        iqm_series = pd.to_numeric(group_df['iqm'], errors='coerce').dropna()
        if iqm_series.empty:
            print(f"      * {group_name}: no IQM values available")
            continue

        avg_iqm = float(iqm_series.mean())
        avg_q25_series = pd.to_numeric(group_df['q25'], errors='coerce').dropna()
        avg_q75_series = pd.to_numeric(group_df['q75'], errors='coerce').dropna()
        avg_q25 = float(avg_q25_series.mean()) if not avg_q25_series.empty else np.nan
        avg_q75 = float(avg_q75_series.mean()) if not avg_q75_series.empty else np.nan

        peak_idx = iqm_series.idxmax()
        peak_row = group_df.loc[peak_idx]
        peak_iqm = float(peak_row['iqm'])
        peak_step = float(peak_row['step'])
        peak_q25 = pd.to_numeric(pd.Series([peak_row.get('q25')]), errors='coerce').iloc[0]
        peak_q75 = pd.to_numeric(pd.Series([peak_row.get('q75')]), errors='coerce').iloc[0]

        final_idx = group_df['step'].idxmax()
        final_row = group_df.loc[final_idx]
        final_iqm = float(final_row['iqm'])
        final_step = float(final_row['step'])
        final_q25 = pd.to_numeric(pd.Series([final_row.get('q25')]), errors='coerce').iloc[0]
        final_q75 = pd.to_numeric(pd.Series([final_row.get('q75')]), errors='coerce').iloc[0]

        avg_interval = _format_interval(avg_iqm, avg_q25, avg_q75)
        final_interval = _format_interval(final_iqm, final_q25, final_q75)
        peak_interval = _format_interval(peak_iqm, peak_q25, peak_q75)

        print(
            "      * "
            f"{group_name}: avg={avg_interval}, final={final_interval} at {final_step:.1f}M steps, "
            f"peak={peak_interval} at {peak_step:.1f}M steps"
        )

        max_runs = None
        if 'n_runs' in group_df:
            run_counts = pd.to_numeric(group_df['n_runs'], errors='coerce').dropna()
            if not run_counts.empty:
                max_runs = int(run_counts.max())

        summaries.append(
            {
                'metric': metric,
                'group': group_name,
                'avg_iqm': avg_iqm,
                'avg_q25': avg_q25,
                'avg_q75': avg_q75,
                'peak_iqm': peak_iqm,
                'peak_step': peak_step,
                'peak_q25': peak_q25,
                'peak_q75': peak_q75,
                'final_iqm': final_iqm,
                'final_step': final_step,
                'final_q25': final_q25,
                'final_q75': final_q75,
                'n_points': len(group_df),
                'max_runs': max_runs,
            }
        )

    return summaries


def create_ablation_chart(
    df: pd.DataFrame,
    metric: str,
    title: str = "",
    use_short_labels: bool = True,
    show_iqr: bool = True,
    base_text_size: float = 25.0,
    chart_width: int = 800,
    chart_height: int = 400,
    line_width: float = 3.0,
    y_tick_count: Optional[int] = None,
    sort_by_value: bool = False,
) -> alt.Chart:
    chart_df = create_short_labels(df, use_short_labels)
    group_col = 'short_group' if use_short_labels and 'short_group' in chart_df.columns else 'group'

    axis_label_large = scaled_font_size(base_text_size, 1.5)
    axis_title_large = scaled_font_size(base_text_size, 1.5)
    legend_label_large = scaled_font_size(base_text_size, 1.1)
    legend_title_large = scaled_font_size(base_text_size, 1.5)
    legend_symbol_large = scaled_font_size(base_text_size, 50)
    legend_symbol_stroke = max(2, scaled_font_size(base_text_size, 0.5))
    legend_label_limit = max(220, int(round(base_text_size * 12)))
    chart_title_large = scaled_font_size(base_text_size, 1.6)
    default_padding = {"right": 25}

    axis_kwargs = {
        "labelFontSize": axis_label_large,
        "titleFontSize": axis_title_large,
        "labelFont": ALT_FONT_FAMILY,
        "titleFont": ALT_FONT_FAMILY,
    }
    y_axis_kwargs = dict(axis_kwargs)
    if y_tick_count is not None:
        y_axis_kwargs["tickCount"] = max(2, int(y_tick_count))
    axis_config = alt.Axis(**y_axis_kwargs)

    max_step = float(chart_df['step'].max()) if not chart_df['step'].empty else 0.0
    # Keep the x-axis tight to the observed data so we do not render empty space past the
    # final training step. Altair expands the domain by default, so explicitly disable
    # that behaviour once we know the maximum step.
    if max_step > 0:
        x_scale = alt.Scale(domain=[0, max_step], nice=False)
    else:
        x_scale = alt.Scale(domain=[0, 1.0], nice=True)

    # When sorting by value, always use the original 'group' column to extract numeric values
    # Then map to display column (short_group or group) while preserving sorted order
    if sort_by_value:
        if use_short_labels and 'short_group' in chart_df.columns:
            # Create mapping from group to short_group, keeping only unique mappings
            group_to_display = {}
            for g, sg in zip(chart_df['group'], chart_df['short_group']):
                group_to_display[g] = sg
            # Sort original groups by numeric value
            sorted_groups = sorted(set(chart_df['group']), key=_parameter_value_sort_key)
            # Map to short labels while preserving order
            legend_domain = [group_to_display[g] for g in sorted_groups if g in group_to_display]
        else:
            # Sort by numeric value using the group column
            legend_domain = sorted(set(chart_df[group_col]), key=_parameter_value_sort_key)
    else:
        legend_domain = build_algorithm_legend_domain(chart_df[group_col], sort_by_value=False)
    color_legend = alt.Legend(
        title=None,
        symbolOpacity=1.0,
        symbolType='stroke',
        symbolStrokeWidth=legend_symbol_stroke,
        symbolSize=legend_symbol_large,
        fillColor="rgba(255,255,255,1)",
        strokeColor="gray",
        padding=5,
        cornerRadius=3,
        orient="bottom-left",
        labelFont=ALT_FONT_FAMILY,
        titleFont=ALT_FONT_FAMILY,
        labelFontSize=legend_label_large,
        titleFontSize=legend_title_large,
        labelFontWeight="bold",
        titleFontWeight="bold",
        # labelFontStyle="italic",
        labelPadding=8,
        labelLimit=legend_label_limit,
    )
    color_scale = alt.Scale(domain=legend_domain) if legend_domain else alt.Undefined
    color_encoding = alt.Color(
        f'{group_col}:N',
        legend=color_legend,
        scale=color_scale,
    )

    base = alt.Chart(chart_df).encode(
        x=alt.X(
            'step:Q',
            title='Training Steps (Millions)',
            scale=x_scale,
            axis=alt.Axis(**axis_kwargs),
        )
    )

    metric_axis_title = metric.split('/')[-1].replace('_', ' ').title()

    smoothed_base = base.transform_window(
        frame=[-5, 5], groupby=[group_col],
        smooth_iqm='mean(iqm)', smooth_q25='mean(q25)', smooth_q75='mean(q75)'
    )

    bands = smoothed_base.mark_area(opacity=0.25).encode(
        color=color_encoding,
        y=alt.Y(
            'smooth_q25:Q',
            title=metric_axis_title,
            axis=axis_config,
        ),
        y2=alt.Y2('smooth_q75:Q')
    )

    lines = smoothed_base.mark_line(strokeWidth=line_width).encode(
        color=color_encoding,
        y=alt.Y(
            'smooth_iqm:Q',
            title=metric_axis_title,
            axis=axis_config,
        ),
        tooltip=[alt.Tooltip(f'{group_col}:N', title='Group'),
                alt.Tooltip('step:Q', title='Step (M)', format='.2f'),
                alt.Tooltip('smooth_iqm:Q', title='IQM', format='.3f')]
    )

    chart_layers = bands + lines if show_iqr else lines

    return (
        chart_layers.properties(
            width=chart_width,
            height=chart_height,
            title=alt.TitleParams(
                text=title or f"{metric.replace('_', ' ').title()} Ablation Study",
                font=ALT_FONT_FAMILY,
                fontSize=chart_title_large,
                fontWeight='bold',
            ),
            padding=default_padding,
        )
        .configure_axis(
            grid=True,
            gridOpacity=0.3,
            labelFontSize=axis_label_large,
            titleFontSize=axis_title_large,
            labelFont=ALT_FONT_FAMILY,
            titleFont=ALT_FONT_FAMILY,
        )
        .configure_legend(
        labelFont=ALT_FONT_FAMILY,
        titleFont=ALT_FONT_FAMILY,
        labelFontSize=legend_label_large,
        titleFontSize=legend_title_large,
        labelFontWeight="bold",
        titleFontWeight="bold",
        labelPadding=8,
        labelLimit=legend_label_limit,
    )
        .interactive()
    )


def main(wandb_entity: str, wandb_project: str = "crl_experiments", group: str = "slippery_ant_ccbp_sweep",
         metric: str = "charts/mean_episodic_return", split_by: Optional[str] = None,
         grouping_mode: Literal["parameter", "seed", "config"] = "parameter", top_k: Optional[int] = None,
         ranking_criteria: Literal["final", "peak", "average", "auc"] = "final", short_labels: bool = True,
         plot_title: Optional[str] = None,
         output_dir: str = "./plots/ablations", ext: str = "png", debug: bool = False,
         show_iqr: bool = True,
         show_metric_in_legend: bool = True,
         base_text_size: float = 20.0,
         chart_width: int = 800,
         chart_height: int = 400,
         line_width: float = 6.0,
         y_tick_count: Optional[int] = None,
         created_after: Optional[str] = None,
         scientific_labels: bool = False,
         include_failed: bool = False,
         sort_by_value: bool = False):
    df = fetch_ablation_data(wandb_entity, wandb_project, group, metric, split_by, show_metric_in_legend, created_after, scientific_labels, include_failed)
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

    agg_df = agg_df.sort_values(['group', 'step'])
    title_parts = [group.replace('_', ' ').title()]
    if split_by: title_parts.append(f"by {split_by}")
    if top_k: title_parts.append(f"(top {top_k} by {ranking_criteria})")
    title_parts.append(f"({grouping_mode} mode)")
    title = ": ".join(title_parts)
    if plot_title:
        title = plot_title

    performance_summary = summarize_group_performance(agg_df, metric)
    if performance_summary:
        agg_df.attrs['performance_summary'] = performance_summary
        ranked_records = [rec for rec in performance_summary if np.isfinite(rec['final_iqm'])]
        if ranked_records:
            top_records = sorted(ranked_records, key=lambda record: record['final_iqm'], reverse=True)[:3]
            top_dict = {record['group']: round(record['final_iqm'], 3) for record in top_records}
            print(f"\nTop performers: {top_dict}")
        else:
            print("\nTop performers: {}")
    else:
        final_data = agg_df.groupby('group')['iqm'].last().sort_values(ascending=False)
        print(f"\nTop performers: {dict(final_data.head(3))}")

    chart = create_ablation_chart(
        agg_df,
        metric,
        title,
        short_labels,
        show_iqr=show_iqr,
        base_text_size=base_text_size,
        chart_width=chart_width,
        chart_height=chart_height,
        line_width=line_width,
        y_tick_count=y_tick_count,
        sort_by_value=sort_by_value,
    )

    save_stem = f"{group}_{metric.replace('/', '_')}"
    if split_by: save_stem += f"_by_{split_by}"
    if top_k: save_stem += f"_top{top_k}_{ranking_criteria}"
    save_stem += f"_{grouping_mode}_ablation"
    if plot_title:
        save_stem = slugify_title(plot_title)
    save_name = f"{save_stem}.{ext}"

    save_path = Path(output_dir) / save_name
    save_path.parent.mkdir(exist_ok=True, parents=True)
    chart.save(str(save_path))
    print(f"Chart saved to: {save_path}")
    return chart


if __name__ == "__main__":
    tyro.cli(main)
