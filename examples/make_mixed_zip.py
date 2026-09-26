"""Generate examples/datasets/shapes_survey.zip, a mixed dataset: images plus a table.

shapes_survey/images/<class>/*.jpg holds drawn shapes in three class folders, and
shapes_survey/annotations.csv has one row per annotated image. The two disagree in planted
ways, which group mode should find by linking the table to the files:
- 3 rows point to images that aren't in the zip
- 4 images have no row in the table
- 3 rows give a label that differs from the image's folder
- the table also has its own problems: a duplicate row, missing-value tokens ("N/A"),
  confidence stored as percent strings, and a free-text notes column
"""

from __future__ import annotations

import csv
import io
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from make_shapes_zip import jpeg, shape

OUT = Path(__file__).parent / "datasets" / "shapes_survey.zip"
COUNTS = {"circles": 16, "squares": 12, "triangles": 10}
ANNOTATORS = ["ana", "ben", "chen", "dia"]


def main(seed: int = 5) -> Path:
    rng = random.Random(seed)
    images: dict[str, bytes] = {}
    for label, n in COUNTS.items():
        for i in range(1, n + 1):
            images[f"{label}/{label[:-1]}_{i:03d}.jpg"] = jpeg(shape(label, rng))

    names = sorted(images)
    unannotated = set(rng.sample(names, 4))
    rows = []
    for name in names:
        if name in unannotated:
            continue
        label = name.split("/")[0]
        rows.append(
            {
                "image": name.split("/")[1],  # file name only: linking must match on it
                "label": label[:-1],  # singular ("circle") while folders are plural
                "annotator": rng.choice(ANNOTATORS),
                "confidence": f"{rng.randint(70, 99)}%",
                "area_px": str(rng.randint(400, 9000)),
                "notes": rng.choice(["", "", "clear", "edge of frame", "N/A", "faint outline"]),
            }
        )
    # labels that disagree with the folder
    for row, wrong in zip(rows[2:30:10], ["square", "triangle", "circle"], strict=True):
        row["label"] = wrong if wrong != row["label"] else "square"
    # rows for images that aren't in the zip
    for k in range(3):
        rows.append(
            {
                "image": f"circle_{100 + k:03d}.jpg",
                "label": "circle",
                "annotator": rng.choice(ANNOTATORS),
                "confidence": f"{rng.randint(70, 99)}%",
                "area_px": str(rng.randint(400, 9000)),
                "notes": "re-shoot",
            }
        )
    rows.append(dict(rows[5]))  # an exact duplicate row
    rows[8]["confidence"] = "N/A"
    rows[11]["area_px"] = "N/A"

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "shapes_survey/README.txt",
            "Synthetic shapes with an annotation table, for MOSAIC EDA group mode.\n",
        )
        z.writestr("shapes_survey/annotations.csv", buf.getvalue())
        for name, data in images.items():
            z.writestr(f"shapes_survey/images/{name}", data)
    print(f"Wrote {len(images)} images and {len(rows)} rows to {OUT}")
    return OUT


if __name__ == "__main__":
    main()
