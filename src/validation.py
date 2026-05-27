from pathlib import Path


def build_path_validation_entries(output_dir: Path, toc: dict, output_targets: dict) -> list[dict]:
    return [
        {
            "index": idx,
            "display_path": toc[idx],
            "output_stem": str(output_dir / output_targets[idx]),
        }
        for idx in sorted(toc.keys())
    ]
