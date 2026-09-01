"""Zero-data sanity check for the Anchor Fidelity Curve.

Run with:
    PYTHONPATH=src python examples/toy_afc.py

Both toy systems are perfect at the source camera. The billboard-like system
collapses off-axis while the coherent-world toy degrades slowly.
"""

from __future__ import annotations

import json

from refworld.metrics import summarize_curve


def main() -> None:
    displacement = [0.0, 2.0, 5.0, 10.0, 20.0]
    systems = {
        "source-view-billboard": [1.0, 0.72, 0.38, 0.14, 0.05],
        "locally-coherent-world": [1.0, 0.97, 0.91, 0.82, 0.67],
    }

    report = {}
    for name, scores in systems.items():
        summary = summarize_curve(displacement, scores, failure_threshold=0.5)
        report[name] = {
            "displacements_deg": displacement,
            "similarity": scores,
            **summary.as_dict(),
        }

    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
