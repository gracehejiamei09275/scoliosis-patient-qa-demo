from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .web_app import main


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run() -> None:
    """Start the public demo with synthetic data only."""
    data_file = PROJECT_ROOT / "data" / "sample_scoliosis_data.xlsx"
    output_dir = Path(tempfile.gettempdir()) / "scoliosis_public_outputs"
    port = int(os.environ.get("PORT", "8000"))
    main(
        [
            "--data",
            str(data_file),
            "--out",
            str(output_dir),
            "--llm",
            "mock",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ]
    )


if __name__ == "__main__":
    run()
