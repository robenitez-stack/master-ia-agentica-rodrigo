import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import nbformat
import numpy
import pandas
import sklearn


def collect() -> dict:
    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "libraries": {
            "matplotlib": matplotlib.__version__,
            "nbformat": nbformat.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "device": {"selected": "cpu", "cuda_available": False, "mps_available": False},
    }
    try:
        import torch

        result["libraries"]["torch"] = torch.__version__
        result["device"]["cuda_available"] = torch.cuda.is_available()
        result["device"]["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if result["device"]["cuda_available"]:
            result["device"]["selected"] = "cuda"
        elif result["device"]["mps_available"]:
            result["device"]["selected"] = "mps"
    except ImportError:
        result["libraries"]["torch"] = None
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(collect(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
