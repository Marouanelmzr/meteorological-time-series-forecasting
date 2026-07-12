# This script orchestrates the cleaning of the Arome dataset by executing a series of steps: 
# validating the structure, computing reference coordinates, filtering by distance, and deduplicating the data. 
# Each step is implemented in a separate module, and this script calls them in sequence to produce a final cleaned dataset.
import sys
from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parents[1] 
sys.path.insert(0, str(PROJECT_ROOT))

from src.cleaning import validate_structure, compute_reference_coordinates, filter_by_distance, deduplicate


STEPS = [
    ("1. validate_structure", validate_structure),
    ("2. compute_reference_coordinates", compute_reference_coordinates),
    ("3. filter_by_distance", filter_by_distance),
    ("4. deduplicate", deduplicate),
]


def main():
    for label, step in STEPS:
        print(f"\n{'=' * 60}")
        print(label)
        print("=" * 60)
        step.main()

    print(f"\n{'=' * 60}")
    print("Pipeline done. Final file : Datasetfinal_clean_final.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()