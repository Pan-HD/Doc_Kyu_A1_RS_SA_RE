from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search_space.encoding import encode_architecture, encoding_dimension
from src.search_space.space import enumerate_architectures


def main():
    arches = list(enumerate_architectures())
    unique_arches = set(arches)

    encodings = [tuple(encode_architecture(a).tolist()) for a in arches]

    print(f"Total architectures: {len(arches)}")
    print(f"Unique architectures: {len(unique_arches)}")
    print(f"Encoding dimension: {encoding_dimension()}")
    print(f"Unique encodings: {len(set(encodings))}")

    assert len(arches) == 1728
    assert len(unique_arches) == 1728
    assert len(set(encodings)) == 1728

    print("Search-space check: PASSED")


if __name__ == "__main__":
    main()
