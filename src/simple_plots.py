from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from .config import GROUP_LABELS

WIDTH = 1000
HEIGHT = 700
MARGIN_LEFT = 185
MARGIN_RIGHT = 55
MARGIN_TOP = 70
MARGIN_BOTTOM = 95
PLOT_WIDTH = WIDTH - MARGIN_LEFT - MARGIN_RIGHT
PLOT_HEIGHT = HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

COLORS = {
    "brace_only": (54, 116, 181, 210),
    "brace_training": (205, 84, 84, 210),
    "before_matching": (112, 128, 144, 220),
    "after_matching": (42, 157, 143, 220),
    "actual_predicted": (67, 97, 238, 210),
}


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica.ttc",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _canvas(title: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGBA", (WIDTH, HEIGHT), (255, 255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((MARGIN_LEFT, 24), title, fill=(28, 33, 39, 255), font=_font(24, bold=True))
    return image, draw


def _save(image: Image.Image, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(path)


def _bounds(values: np.ndarray, pad_ratio: float = 0.08) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0
    low = float(np.min(values))
    high = float(np.max(values))
    if low == high:
        return low - 1, high + 1
    pad = (high - low) * pad_ratio
    return low - pad, high + pad


def _x(value: float, low: float, high: float) -> int:
    return int(MARGIN_LEFT + (value - low) / (high - low) * PLOT_WIDTH)


def _y(value: float, low: float, high: float) -> int:
    return int(MARGIN_TOP + PLOT_HEIGHT - (value - low) / (high - low) * PLOT_HEIGHT)


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    x_label: str,
    y_label: str,
    x_low: float,
    x_high: float,
    y_low: float,
    y_high: float,
) -> None:
    axis_color = (70, 78, 88, 255)
    grid_color = (226, 231, 236, 255)
    draw.line(
        [(MARGIN_LEFT, MARGIN_TOP), (MARGIN_LEFT, MARGIN_TOP + PLOT_HEIGHT)],
        fill=axis_color,
        width=2,
    )
    draw.line(
        [(MARGIN_LEFT, MARGIN_TOP + PLOT_HEIGHT), (MARGIN_LEFT + PLOT_WIDTH, MARGIN_TOP + PLOT_HEIGHT)],
        fill=axis_color,
        width=2,
    )

    label_font = _font(16)
    tick_font = _font(13)
    for i in range(6):
        y_val = y_low + (y_high - y_low) * i / 5
        y_pos = _y(y_val, y_low, y_high)
        draw.line(
            [(MARGIN_LEFT, y_pos), (MARGIN_LEFT + PLOT_WIDTH, y_pos)],
            fill=grid_color,
            width=1,
        )
        draw.text((24, y_pos - 8), f"{y_val:.1f}", fill=(80, 86, 96, 255), font=tick_font)

    for i in range(6):
        x_val = x_low + (x_high - x_low) * i / 5
        x_pos = _x(x_val, x_low, x_high)
        draw.line(
            [(x_pos, MARGIN_TOP + PLOT_HEIGHT), (x_pos, MARGIN_TOP + PLOT_HEIGHT + 6)],
            fill=axis_color,
            width=1,
        )
        draw.text((x_pos - 18, MARGIN_TOP + PLOT_HEIGHT + 12), f"{x_val:.1f}", fill=(80, 86, 96, 255), font=tick_font)

    draw.text((MARGIN_LEFT + PLOT_WIDTH // 2 - 55, HEIGHT - 46), x_label, fill=axis_color, font=label_font)
    draw.text((16, MARGIN_TOP + PLOT_HEIGHT // 2 - 15), y_label, fill=axis_color, font=label_font)


def _draw_categorical_axes(
    draw: ImageDraw.ImageDraw,
    x_label: str,
    y_label: str,
    y_low: float,
    y_high: float,
) -> None:
    axis_color = (70, 78, 88, 255)
    grid_color = (226, 231, 236, 255)
    draw.line(
        [(MARGIN_LEFT, MARGIN_TOP), (MARGIN_LEFT, MARGIN_TOP + PLOT_HEIGHT)],
        fill=axis_color,
        width=2,
    )
    draw.line(
        [(MARGIN_LEFT, MARGIN_TOP + PLOT_HEIGHT), (MARGIN_LEFT + PLOT_WIDTH, MARGIN_TOP + PLOT_HEIGHT)],
        fill=axis_color,
        width=2,
    )
    tick_font = _font(13)
    for i in range(6):
        y_val = y_low + (y_high - y_low) * i / 5
        y_pos = _y(y_val, y_low, y_high)
        draw.line(
            [(MARGIN_LEFT, y_pos), (MARGIN_LEFT + PLOT_WIDTH, y_pos)],
            fill=grid_color,
            width=1,
        )
        draw.text((24, y_pos - 8), f"{y_val:.1f}", fill=(80, 86, 96, 255), font=tick_font)
    label_font = _font(16)
    draw.text((MARGIN_LEFT + PLOT_WIDTH // 2 - 55, HEIGHT - 46), x_label, fill=axis_color, font=label_font)
    draw.text((16, MARGIN_TOP + PLOT_HEIGHT // 2 - 15), y_label, fill=axis_color, font=label_font)


def _legend(draw: ImageDraw.ImageDraw, labels: list[str], keys: list[str]) -> None:
    x0 = WIDTH - 255
    y0 = 30
    font = _font(14)
    for i, (label, key) in enumerate(zip(labels, keys)):
        y = y0 + i * 24
        draw.rectangle([x0, y + 4, x0 + 16, y + 20], fill=COLORS[key], outline=(40, 40, 40, 255))
        draw.text((x0 + 24, y + 2), label, fill=(45, 52, 60, 255), font=font)


def save_histogram_by_group(
    df: pd.DataFrame,
    column: str,
    output_path: Path | str,
    title: str,
    x_label: str,
    bins: int = 10,
) -> None:
    image, draw = _canvas(title)
    values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
    x_low, x_high = _bounds(values, pad_ratio=0.02)
    bin_edges = np.linspace(x_low, x_high, bins + 1)
    group_keys = ["brace_only", "brace_training"]
    counts_by_group = []
    for group in group_keys:
        group_values = pd.to_numeric(
            df.loc[df["treatment_group"].eq(group), column], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        counts, _ = np.histogram(group_values, bins=bin_edges)
        counts_by_group.append(counts)
    max_count = max(int(np.max(counts)) for counts in counts_by_group) or 1
    _draw_axes(draw, x_label, "Count", x_low, x_high, 0, max_count * 1.15)

    bin_width_px = PLOT_WIDTH / bins
    bar_width = bin_width_px / (len(group_keys) + 1)
    for bin_idx in range(bins):
        for group_idx, group in enumerate(group_keys):
            count = counts_by_group[group_idx][bin_idx]
            x0 = int(MARGIN_LEFT + bin_idx * bin_width_px + group_idx * bar_width + 4)
            x1 = int(x0 + bar_width - 5)
            y0 = _y(float(count), 0, max_count * 1.15)
            y1 = MARGIN_TOP + PLOT_HEIGHT
            draw.rectangle([x0, y0, x1, y1], fill=COLORS[group], outline=(255, 255, 255, 255))

    _legend(draw, [GROUP_LABELS[k] for k in group_keys], group_keys)
    _save(image, output_path)


def save_boxplot_by_group(
    df: pd.DataFrame,
    column: str,
    output_path: Path | str,
    title: str,
    y_label: str,
) -> None:
    image, draw = _canvas(title)
    group_keys = ["brace_only", "brace_training"]
    all_values = pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(dtype=float)
    y_low, y_high = _bounds(all_values)
    _draw_axes(draw, "Treatment group", y_label, 0, 3, y_low, y_high)

    font = _font(15)
    for i, group in enumerate(group_keys, start=1):
        values = pd.to_numeric(
            df.loc[df["treatment_group"].eq(group), column], errors="coerce"
        ).dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        low, high = np.min(values), np.max(values)
        x_center = _x(i, 0, 3)
        box_half = 58
        draw.line([(x_center, _y(low, y_low, y_high)), (x_center, _y(high, y_low, y_high))], fill=COLORS[group], width=3)
        draw.rectangle(
            [x_center - box_half, _y(q3, y_low, y_high), x_center + box_half, _y(q1, y_low, y_high)],
            fill=COLORS[group],
            outline=(40, 45, 52, 255),
            width=2,
        )
        draw.line(
            [(x_center - box_half, _y(median, y_low, y_high)), (x_center + box_half, _y(median, y_low, y_high))],
            fill=(32, 35, 40, 255),
            width=3,
        )
        label = GROUP_LABELS[group]
        draw.text((x_center - 70, HEIGHT - 70), label, fill=(55, 62, 70, 255), font=font)

    _save(image, output_path)


def save_scatter_by_group(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    output_path: Path | str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    image, draw = _canvas(title)
    x_values = pd.to_numeric(df[x_column], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(df[y_column], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_low, x_high = _bounds(x_values[valid])
    y_low, y_high = _bounds(y_values[valid])
    _draw_axes(draw, x_label, y_label, x_low, x_high, y_low, y_high)

    for group in ["brace_only", "brace_training"]:
        part = df[df["treatment_group"].eq(group)]
        for _, row in part.iterrows():
            x_val = row[x_column]
            y_val = row[y_column]
            if pd.isna(x_val) or pd.isna(y_val):
                continue
            x_pos = _x(float(x_val), x_low, x_high)
            y_pos = _y(float(y_val), y_low, y_high)
            draw.ellipse([x_pos - 5, y_pos - 5, x_pos + 5, y_pos + 5], fill=COLORS[group])

    _legend(draw, [GROUP_LABELS["brace_only"], GROUP_LABELS["brace_training"]], ["brace_only", "brace_training"])
    _save(image, output_path)


def save_smd_balance_plot(balance: pd.DataFrame, output_path: Path | str) -> None:
    image, draw = _canvas("Covariate balance before vs after PSM")
    variables = list(dict.fromkeys(balance["variable"].tolist()))
    x_low, x_high = 0, max(1.0, float(balance["abs_smd"].max()) * 1.15)
    y_low, y_high = -0.5, len(variables) - 0.5

    axis_color = (70, 78, 88, 255)
    draw.line([(MARGIN_LEFT, MARGIN_TOP), (MARGIN_LEFT, MARGIN_TOP + PLOT_HEIGHT)], fill=axis_color, width=2)
    draw.line([(MARGIN_LEFT, MARGIN_TOP + PLOT_HEIGHT), (MARGIN_LEFT + PLOT_WIDTH, MARGIN_TOP + PLOT_HEIGHT)], fill=axis_color, width=2)
    threshold_x = _x(0.1, x_low, x_high)
    draw.line([(threshold_x, MARGIN_TOP), (threshold_x, MARGIN_TOP + PLOT_HEIGHT)], fill=(219, 119, 57, 255), width=2)
    draw.text((threshold_x + 5, MARGIN_TOP + 8), "SMD=0.10", fill=(140, 80, 30, 255), font=_font(13))

    font = _font(14)
    bar_height = max(16, int(PLOT_HEIGHT / max(len(variables), 1) / 4))
    for i, variable in enumerate(variables):
        y_center = _y(i, y_low, y_high)
        draw.text((16, y_center - 8), variable, fill=(55, 62, 70, 255), font=font)
        for offset, stage in [(-bar_height, "before_matching"), (bar_height, "after_matching")]:
            row = balance[(balance["variable"].eq(variable)) & (balance["stage"].eq(stage))]
            if row.empty:
                continue
            value = float(row.iloc[0]["abs_smd"])
            x1 = _x(value, x_low, x_high)
            draw.rectangle(
                [MARGIN_LEFT, y_center + offset - bar_height // 2, x1, y_center + offset + bar_height // 2],
                fill=COLORS[stage],
            )
    for i in range(6):
        x_val = x_low + (x_high - x_low) * i / 5
        x_pos = _x(x_val, x_low, x_high)
        draw.text((x_pos - 14, HEIGHT - 70), f"{x_val:.2f}", fill=(80, 86, 96, 255), font=_font(13))
    draw.text((MARGIN_LEFT + PLOT_WIDTH // 2 - 50, HEIGHT - 45), "Absolute SMD", fill=axis_color, font=_font(16))
    _legend(draw, ["Before matching", "After matching"], ["before_matching", "after_matching"])
    _save(image, output_path)


def save_prediction_scatter(predictions: pd.DataFrame, output_path: Path | str) -> None:
    image, draw = _canvas("Cross-validated Ridge predictions")
    x_values = predictions["actual_cobb_change"].to_numpy(dtype=float)
    y_values = predictions["predicted_cobb_change"].to_numpy(dtype=float)
    low, high = _bounds(np.concatenate([x_values, y_values]))
    _draw_axes(draw, "Actual Cobb change", "Predicted Cobb change", low, high, low, high)
    draw.line([(_x(low, low, high), _y(low, low, high)), (_x(high, low, high), _y(high, low, high))], fill=(90, 90, 90, 255), width=2)

    for actual, predicted in zip(x_values, y_values):
        x_pos = _x(float(actual), low, high)
        y_pos = _y(float(predicted), low, high)
        draw.ellipse([x_pos - 5, y_pos - 5, x_pos + 5, y_pos + 5], fill=COLORS["actual_predicted"])
    _save(image, output_path)


def save_binary_rate_plot(binary_analysis: pd.DataFrame, output_path: Path | str) -> None:
    image, draw = _canvas("Improved >=5deg rate by treatment group")
    group_rates = binary_analysis[binary_analysis["analysis"].eq("group_rate")].copy()
    group_rates["treatment_group"] = group_rates["estimand"].str.extract(r"in (.+)$")
    group_keys = ["brace_only", "brace_training"]
    y_low, y_high = 0.0, 1.0
    _draw_categorical_axes(draw, "Treatment group", "Improved >=5deg rate", y_low, y_high)

    font = _font(15)
    x_positions = [
        int(MARGIN_LEFT + PLOT_WIDTH * 0.33),
        int(MARGIN_LEFT + PLOT_WIDTH * 0.67),
    ]
    for x_center, group in zip(x_positions, group_keys):
        row = group_rates[group_rates["treatment_group"].eq(group)]
        if row.empty:
            continue
        rate = float(row.iloc[0]["estimate"])
        n_obs = int(row.iloc[0]["n_obs"])
        events = int(row.iloc[0]["events"])
        bar_width = 110
        draw.rectangle(
            [x_center - bar_width // 2, _y(rate, y_low, y_high), x_center + bar_width // 2, MARGIN_TOP + PLOT_HEIGHT],
            fill=COLORS[group],
            outline=(40, 45, 52, 255),
        )
        draw.text((x_center - 42, _y(rate, y_low, y_high) - 26), f"{rate:.1%}", fill=(32, 35, 40, 255), font=font)
        draw.text((x_center - 52, HEIGHT - 70), GROUP_LABELS[group], fill=(55, 62, 70, 255), font=font)
        draw.text((x_center - 34, HEIGHT - 48), f"{events}/{n_obs}", fill=(80, 86, 96, 255), font=_font(13))
    _save(image, output_path)


def save_psm_sensitivity_plot(sensitivity: pd.DataFrame, output_path: Path | str) -> None:
    image, draw = _canvas("PSM sensitivity across calipers")
    work = sensitivity.copy()
    work["x_position"] = np.arange(len(work))
    y_values = pd.to_numeric(work["att_cobb_change"], errors="coerce").to_numpy(dtype=float)
    ci_low = pd.to_numeric(work["att_ci_lower"], errors="coerce").to_numpy(dtype=float)
    ci_high = pd.to_numeric(work["att_ci_upper"], errors="coerce").to_numpy(dtype=float)
    y_low, y_high = _bounds(np.concatenate([y_values, ci_low, ci_high]))
    _draw_categorical_axes(draw, "Caliper setting", "ATT Cobb change", y_low, y_high)

    zero_y = _y(0, y_low, y_high)
    if MARGIN_TOP <= zero_y <= MARGIN_TOP + PLOT_HEIGHT:
        draw.line([(MARGIN_LEFT, zero_y), (MARGIN_LEFT + PLOT_WIDTH, zero_y)], fill=(120, 120, 120, 255), width=2)

    font = _font(12)
    n_points = len(work)
    if n_points <= 1:
        x_positions = [MARGIN_LEFT + PLOT_WIDTH // 2]
    else:
        inner_pad = 65
        x_positions = [
            int(MARGIN_LEFT + inner_pad + i * (PLOT_WIDTH - 2 * inner_pad) / (n_points - 1))
            for i in range(n_points)
        ]
    for x_pos, (_, row) in zip(x_positions, work.iterrows()):
        estimate = float(row["att_cobb_change"]) if pd.notna(row["att_cobb_change"]) else np.nan
        low = float(row["att_ci_lower"]) if pd.notna(row["att_ci_lower"]) else np.nan
        high = float(row["att_ci_upper"]) if pd.notna(row["att_ci_upper"]) else np.nan
        if np.isfinite(low) and np.isfinite(high):
            draw.line([(x_pos, _y(low, y_low, y_high)), (x_pos, _y(high, y_low, y_high))], fill=(50, 60, 70, 255), width=3)
        if np.isfinite(estimate):
            y_pos = _y(estimate, y_low, y_high)
            draw.ellipse([x_pos - 7, y_pos - 7, x_pos + 7, y_pos + 7], fill=(42, 157, 143, 255), outline=(25, 30, 35, 255))
        label = str(row["caliper_label"]).replace("_sd_logit", " sd")
        draw.text((x_pos - 38, HEIGHT - 72), label, fill=(70, 78, 88, 255), font=font)
        draw.text((x_pos - 20, HEIGHT - 52), f"n={int(row['matched_pairs_outcome_complete'])}", fill=(90, 96, 104, 255), font=font)
    _save(image, output_path)
