from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import SHEET_TO_GROUP


@dataclass(frozen=True)
class WorkbookMetadata:
    path: Path
    sheets: list[str]
    rows_by_sheet: dict[str, int]
    columns_by_sheet: dict[str, list[str]]


def inspect_workbook(path: Path | str) -> WorkbookMetadata:
    """Return non-mutating workbook metadata for audit and reporting."""
    workbook_path = Path(path)
    excel = pd.ExcelFile(workbook_path)
    rows_by_sheet: dict[str, int] = {}
    columns_by_sheet: dict[str, list[str]] = {}
    for sheet in excel.sheet_names:
        frame = pd.read_excel(workbook_path, sheet_name=sheet)
        rows_by_sheet[sheet] = int(len(frame))
        columns_by_sheet[sheet] = [str(column) for column in frame.columns]
    return WorkbookMetadata(
        path=workbook_path,
        sheets=[str(sheet) for sheet in excel.sheet_names],
        rows_by_sheet=rows_by_sheet,
        columns_by_sheet=columns_by_sheet,
    )


def read_raw_workbook(path: Path | str) -> pd.DataFrame:
    """Read the configured treatment sheets and stack them into one raw table."""
    workbook_path = Path(path)
    missing_sheets: list[str] = []
    frames: list[pd.DataFrame] = []
    excel = pd.ExcelFile(workbook_path)

    for sheet_name, treatment_group in SHEET_TO_GROUP.items():
        if sheet_name not in excel.sheet_names:
            missing_sheets.append(sheet_name)
            continue
        sheet = pd.read_excel(workbook_path, sheet_name=sheet_name)
        sheet = sheet.drop(columns=["序号.1"], errors="ignore")
        sheet["source_sheet"] = sheet_name
        sheet["treatment_group"] = treatment_group
        frames.append(sheet)

    if missing_sheets:
        raise ValueError(f"Missing required workbook sheet(s): {missing_sheets}")
    if not frames:
        raise ValueError("No treatment sheets could be read from the workbook.")
    return pd.concat(frames, ignore_index=True, sort=False)
