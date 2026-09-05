"""Quick diagnostic: check label range vs class count."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.dataset import discover_classes

class_folders, class_to_idx = discover_classes("landmarks")

print(f"Folder count: {len(class_folders)}")
print(f"Unique names (class_to_idx size): {len(class_to_idx)}")

# Find duplicate names
from collections import Counter
names = [cf.name for cf in class_folders]
dupes = {n: c for n, c in Counter(names).items() if c > 1}
if dupes:
    print(f"\nDUPLICATE folder names ({len(dupes)}):")
    for n, c in sorted(dupes.items()):
        print(f"  '{n}': {c} folders")

# Max label
labels = [class_to_idx[cf.name] for cf in class_folders]
print(f"\nMax label: {max(labels)}")
print(f"num_classes would be: {len(class_to_idx)}")

if max(labels) >= len(class_to_idx):
    print("ERROR: max label >= num_classes!")
else:
    print("Labels look consistent.")
