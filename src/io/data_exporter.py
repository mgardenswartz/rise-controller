import pickle
from pathlib import Path

import jax

def export_to_pickle(
    data: dict[str, jax.Array],
    output_dir: Path,
    filename: str
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / filename
    with file_path.open("wb") as f:
        pickle.dump(data, f)