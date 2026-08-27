from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


WIDTH = 960
HEIGHT = 600
MARGIN_LEFT = 82
MARGIN_RIGHT = 34
MARGIN_TOP = 72
MARGIN_BOTTOM = 72
COLORS = {
    "brace_only": (52, 95, 151),
    "brace_training": (39, 135, 114),
    "accent": (196, 93, 57),
    "grid": (224, 228, 233),
    "text": (35, 40, 48),
    "muted": (101, 111, 126),
}


def _font(size: int = 16) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("Arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    draw.text((MARGIN_LEFT, 26), title, fill=COLORS["text"], font=_font(24))
    return image, draw


def _plot_area() -> tuple[int, int, int, int]:
    return (MARGIN_LEFT, MARGIN_TOP, WIDTH - MARGIN_RIGHT, HEIGHT - MARGIN_BOTTOM)


def _draw_axes(draw: ImageDraw.ImageDraw, y_min: float, y_max: float, y_label: str = "") -> None:
    x0, y0, x1, y1 = _plot_area()
    draw.line((x0, y1, x1, y1), fill=COLORS["text"], width=2)
    draw.line((x0, y0, x0, y1), fill=COLORS["text"], width=2)
    if y_min == y_max:
        y_max = y_min + 1
    for i in range(5):
        y = y1 - (y1 - y0) * i / 4
        value = y_min + (y_max - y_min) * i / 4
        draw.line((x0, y, x1, y), fill=COLORS["grid"], width=1)
        draw.text((12, y - 8), f"{value:.1f}", fill=COLORS["muted"], font=_font(13))
    if y_label:
        draw.text((MARGIN_LEFT, HEIGHT - 42), y_label, fill=COLORS["muted"], font=_font(14))


def _scale_y(value: float, y_min: float, y_max: float) -> float:
    x0, y0, x1, y1 = _plot_area()
    if y_min == y_max:
        y_max = y_min + 1
    return y1 - (value - y_min) / (y_max - y_min) * (y1 - y0)


def save_histogram_by_group(
    df: pd.DataFrame,
    variable: str,
    path: Path,
    title: str,
    bins: int = 8,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = _canvas(title)
    values = pd.to_numeric(df[variable], errors="coerce").dropna()
    if values.empty:
        draw.text((MARGIN_LEFT, MARGIN_TOP), "No observed values", fill=COLORS["muted"], font=_font(18))
        image.save(path)
        return path
    edges = np.linspace(values.min(), values.max(), bins + 1)
    counts_by_group: dict[str, np.ndarray] = {}
    for group, group_df in df.groupby("treatment_group"):
        group_values = pd.to_numeric(group_df[variable], errors="coerce").dropna()
        counts_by_group[group] = np.histogram(group_values, bins=edges)[0]
    max_count = max(int(counts.max()) for counts in counts_by_group.values()) or 1
    _draw_axes(draw, 0, max_count, variable)
    x0, y0, x1, y1 = _plot_area()
    bin_width = (x1 - x0) / bins
    group_keys = list(counts_by_group)
    for i in range(bins):
        for j, group in enumerate(group_keys):
            bar_w = bin_width / max(len(group_keys), 1) * 0.76
            left = x0 + i * bin_width + j * (bin_width / len(group_keys)) + 5
            right = left + bar_w
            top = _scale_y(float(counts_by_group[group][i]), 0, max_count)
            draw.rectangle((left, top, right, y1), fill=COLORS.get(group, COLORS["accent"]))
        label = f"{edges[i]:.0f}"
        draw.text((x0 + i * bin_width + 4, y1 + 10), label, fill=COLORS["muted"], font=_font(12))
    _draw_legend(draw, group_keys)
    image.save(path)
    return path


def _draw_legend(draw: ImageDraw.ImageDraw, groups: list[str]) -> None:
    labels = {"brace_only": "Brace only", "brace_training": "Brace + training"}
    x = WIDTH - 290
    y = 30
    for group in groups:
        draw.rectangle((x, y, x + 16, y + 16), fill=COLORS.get(group, COLORS["accent"]))
        draw.text((x + 24, y - 1), labels.get(group, group), fill=COLORS["text"], font=_font(14))
        y += 22


def save_group_mean_bar(df: pd.DataFrame, variable: str, path: Path, title: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = _canvas(title)
    complete = df.dropna(subset=[variable])
    means = complete.groupby("treatment_group")[variable].mean()
    if means.empty:
        image.save(path)
        return path
    y_min = min(0.0, float(means.min()) - 1)
    y_max = float(means.max()) + 1
    _draw_axes(draw, y_min, y_max, variable)
    x0, y0, x1, y1 = _plot_area()
    labels = {"brace_only": "Brace only", "brace_training": "Brace + training"}
    spacing = (x1 - x0) / (len(means) + 1)
    bar_w = 120
    for i, (group, mean) in enumerate(means.items(), start=1):
        center = x0 + spacing * i
        top = _scale_y(float(mean), y_min, y_max)
        baseline = _scale_y(0, y_min, y_max)
        draw.rectangle((center - bar_w / 2, top, center + bar_w / 2, baseline), fill=COLORS.get(group, COLORS["accent"]))
        draw.text((center - 56, y1 + 12), labels.get(group, group), fill=COLORS["text"], font=_font(14))
        draw.text((center - 24, top - 24), f"{mean:.2f}", fill=COLORS["text"], font=_font(14))
    image.save(path)
    return path


def save_scatter_by_group(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    path: Path,
    title: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = _canvas(title)
    work = df[[x_col, y_col, "treatment_group"]].dropna()
    if work.empty:
        image.save(path)
        return path
    x_min, x_max = float(work[x_col].min()), float(work[x_col].max())
    y_min, y_max = float(work[y_col].min()), float(work[y_col].max())
    pad_y = max((y_max - y_min) * 0.1, 1)
    y_min -= pad_y
    y_max += pad_y
    _draw_axes(draw, y_min, y_max, x_col)
    x0, y0, x1, y1 = _plot_area()
    if x_min == x_max:
        x_max = x_min + 1
    for _, row in work.iterrows():
        x = x0 + (float(row[x_col]) - x_min) / (x_max - x_min) * (x1 - x0)
        y = _scale_y(float(row[y_col]), y_min, y_max)
        color = COLORS.get(str(row["treatment_group"]), COLORS["accent"])
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    _draw_legend(draw, list(work["treatment_group"].drop_duplicates()))
    image.save(path)
    return path


def save_psm_balance_plot(balance: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = _canvas("PSM balance: absolute standardized mean differences")
    work = balance[balance["stage"].isin(["before_matching", "after_matching"])].copy()
    if work.empty:
        image.save(path)
        return path
    max_value = max(float(work["abs_smd"].max()), 0.1)
    _draw_axes(draw, 0, max_value, "covariate")
    x0, y0, x1, y1 = _plot_area()
    variables = list(work["variable"].drop_duplicates())
    spacing = (x1 - x0) / max(len(variables), 1)
    for i, variable in enumerate(variables):
        subset = work[work["variable"].eq(variable)]
        for j, stage in enumerate(["before_matching", "after_matching"]):
            value = subset.loc[subset["stage"].eq(stage), "abs_smd"]
            if value.empty:
                continue
            left = x0 + i * spacing + 18 + j * 30
            top = _scale_y(float(value.iloc[0]), 0, max_value)
            color = COLORS["accent"] if stage == "before_matching" else COLORS["brace_training"]
            draw.rectangle((left, top, left + 24, y1), fill=color)
        draw.text((x0 + i * spacing + 5, y1 + 10), variable[:10], fill=COLORS["muted"], font=_font(11))
    draw.text((WIDTH - 270, 32), "orange=before  green=after", fill=COLORS["muted"], font=_font(14))
    image.save(path)
    return path


def save_prediction_scatter(predictions: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image, draw = _canvas("Cross-validated predictions")
    work = predictions[["actual_cobb_change", "predicted_cobb_change", "treatment_group"]].dropna()
    if work.empty:
        image.save(path)
        return path
    all_values = pd.concat([work["actual_cobb_change"], work["predicted_cobb_change"]])
    y_min = float(all_values.min()) - 1
    y_max = float(all_values.max()) + 1
    _draw_axes(draw, y_min, y_max, "actual Cobb change")
    x0, y0, x1, y1 = _plot_area()
    draw.line((x0, y1, x1, y0), fill=COLORS["grid"], width=2)
    if y_min == y_max:
        y_max = y_min + 1
    for _, row in work.iterrows():
        x = x0 + (float(row["actual_cobb_change"]) - y_min) / (y_max - y_min) * (x1 - x0)
        y = _scale_y(float(row["predicted_cobb_change"]), y_min, y_max)
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=COLORS.get(str(row["treatment_group"]), COLORS["accent"]))
    _draw_legend(draw, list(work["treatment_group"].drop_duplicates()))
    image.save(path)
    return path


def generate_figures(
    cleaned: pd.DataFrame,
    psm_balance: pd.DataFrame,
    model_predictions: pd.DataFrame,
    figure_dir: Path,
) -> list[Path]:
    complete = cleaned[cleaned["outcome_observed"].eq(1)].copy()
    paths = [
        save_histogram_by_group(cleaned, "age", figure_dir / "age_distribution.png", "Age distribution"),
        save_histogram_by_group(cleaned, "pre_cobb", figure_dir / "pre_cobb_distribution.png", "Baseline Cobb distribution"),
        save_group_mean_bar(complete, "cobb_change", figure_dir / "cobb_change_by_group.png", "Mean Cobb change by group"),
        save_scatter_by_group(complete, "intervention_months", "cobb_change", figure_dir / "duration_vs_cobb_change.png", "Duration vs Cobb change"),
        save_psm_balance_plot(psm_balance, figure_dir / "psm_balance_smd.png"),
        save_prediction_scatter(model_predictions, figure_dir / "model_predictions.png"),
    ]
    return paths
