"""Per-data-type adapters. The Flow's agent steps, guardrails, and review loop are shared;
each adapter supplies what differs: loading, profiling, the cleaning catalog, the crew's
prompts, applying a plan, and report extras."""

from __future__ import annotations

import base64
import io
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from mosaic.crews.image.crew import ImageCrew
from mosaic.crews.table.crew import TableCrew
from mosaic.flow.runtime import JobRuntime
from mosaic.images import pipeline_helpers as image_helpers
from mosaic.images.ops import IMAGE_OPS, image_namespace
from mosaic.images.pipeline_helpers import build_table, export_images
from mosaic.images.profile import ImageProfile, profile_images
from mosaic.images.vision import vision_review
from mosaic.ingest.models import FileManifest, Modality
from mosaic.ingest.sampler import stratified_sample
from mosaic.tables.cleaning import (
    PlanRun,
    conservative_plan,
    distribution_shifts,
    execute_plan,
    export_clean,
    helpers_source,
    pipeline_script,
)
from mosaic.tables.load import LoadedTable, load_table
from mosaic.tables.ops import CleaningOp, CleaningPlan, catalog_text
from mosaic.tables.profile import ProfileResult, profile_table

MAX_CLASS_LOSS = 0.5


class Adapter:
    modality = ""
    unit = "items"
    crew_cls: type = TableCrew

    def __init__(self, rt: JobRuntime) -> None:
        self.rt = rt
        self.notes: list[str] = []
        self.chart_ids: list[str] = []

    # load(manifest) -> str, profile_raw(goal) -> (quality, target, n_items),
    # guard_df, columns, execute(plan), post_checks(run), catalog(), conservative_plan(),
    # apply(plan, run) -> (outputs, quality_clean, n_after), report_extras()


# ---------------------------------------------------------------- tables


class TableAdapter(Adapter):
    modality = "table"
    unit = "rows"
    crew_cls = TableCrew

    def load(self, manifest: FileManifest) -> str:
        tables = sorted(manifest.files_of(Modality.TABLE), key=lambda f: -f.size)
        if len(tables) > 1:
            self.notes.append(
                f"Found {len(tables)} tables; analyzing the largest, '{tables[0].path}'."
            )
        self.table: LoadedTable = load_table(Path(manifest.root) / tables[0].path, tables[0].format)
        self.notes.extend(self.table.notes)
        return f"{tables[0].path}: {len(self.table.df):,} rows x {self.table.df.shape[1]} columns"

    def profile_raw(self, goal: str) -> tuple[float, str | None, int]:
        self.raw: ProfileResult = profile_table(self.table, self.rt.store, goal=goal, stage="raw")
        self.chart_ids = list(self.raw.chart_ids)
        return self.raw.quality, self.raw.target, len(self.table.df)

    @property
    def guard_df(self) -> pd.DataFrame:
        return self.table.df

    @property
    def columns(self) -> list[str]:
        return list(self.table.df.columns)

    def execute(self, plan: CleaningPlan) -> PlanRun:
        return execute_plan(plan, self.table.df, self.rt.store)

    def post_checks(self, run: PlanRun) -> list[str]:
        return distribution_shifts(self.table.df, run)

    def catalog(self) -> str:
        return catalog_text()

    def conservative_plan(self) -> CleaningPlan:
        return conservative_plan(self.raw.types)

    def apply(self, plan: CleaningPlan, run: PlanRun) -> tuple[dict[str, str], float, int]:
        clean = LoadedTable(
            df=run.df.astype("str"), source=self.table.source, format=self.table.format
        )
        self.clean = profile_table(clean, self.rt.store, stage="clean")
        self.chart_ids += self.clean.chart_ids
        out = self.rt.ws.out
        export_clean(run.df, out / "cleaned.csv")
        (out / "cleaning_pipeline.py").write_text(
            pipeline_script(plan, run, self.table), encoding="utf-8"
        )
        outputs = {
            "cleaned": str(out / "cleaned.csv"),
            "pipeline": str(out / "cleaning_pipeline.py"),
        }
        return outputs, self.clean.quality, len(run.df)

    def report_extras(self) -> dict[str, Any]:
        columns = self.rt.store.get(self.raw.artifact_ids[1]).data["columns"]
        return {"columns": columns}

    def gallery(self) -> list[tuple[str, str]]:
        return []


# ---------------------------------------------------------------- images


def _thumb_b64(path: Path, size: int = 96) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def image_pipeline_script(plan: CleaningPlan, run: PlanRun, source: str) -> str:
    steps = []
    for s in run.steps:
        comment = f"    # Step {s.index}: {s.op} [{s.risk}] - {s.rationale}".replace("\n", " ")
        steps.append(comment + "\n" + "\n".join("    " + ln for ln in s.code.splitlines()))
    body = "\n\n".join(steps) or "    pass"
    return f'''"""cleaning_pipeline.py - generated by MOSAIC EDA on {time.strftime("%Y-%m-%d")}.

Source: {source}
Plan: {plan.summary}

Rerun it on the full image folder (class = first folder level):
    python cleaning_pipeline.py path/to/images cleaned_images
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

{helpers_source(image_helpers)}


def clean(df):
{body}
    return df


if __name__ == "__main__":
    source = Path(sys.argv[1] if len(sys.argv) > 1 else "images")
    target = Path(sys.argv[2] if len(sys.argv) > 2 else "cleaned_images")
    df = clean(build_table(source))
    export_images(df, source, target)
    print(f"Saved {{len(df)}} images to", target)
'''


class ImageAdapter(Adapter):
    modality = "image"
    unit = "images"
    crew_cls = ImageCrew

    def load(self, manifest: FileManifest) -> str:
        files = manifest.files_of(Modality.IMAGE)
        self.root = Path(manifest.root)
        self.class_counts = dict(Counter(f.group for f in files))
        self.total = len(files)
        sample = stratified_sample(files, Modality.IMAGE, self.rt.settings.max_sampled_files)
        if sample.is_sample:
            self.notes.append(
                f"Analyzed a stratified sample of {len(sample.files)} of "
                f"{self.total} images (by folder)."
            )
        if not manifest.folders_as_labels:
            self.notes.append(
                "Images aren't organized in class folders, so there's no class "
                "balance or mislabel check."
            )
        # Zips often wrap everything in one folder ("shapes/circles/..."): work inside it, so
        # paths match what the exported pipeline sees when it's run on that folder
        tops = {f.path.split("/")[0] for f in files}
        if len(tops) == 1 and all(f.path.count("/") >= 2 for f in files):
            wrapper = tops.pop()
            self.root = self.root / wrapper
            rel = [(f.path[len(wrapper) + 1 :], f.group) for f in sample.files]
        else:
            rel = [(f.path, f.group) for f in sample.files]
        self.df = build_table(self.root, rel)
        return f"{self.total} images in {len([c for c in self.class_counts if c])} classes" + (
            f" ({len(sample.files)} sampled)" if sample.is_sample else ""
        )

    def profile_raw(self, goal: str) -> tuple[float, str | None, int]:
        self.raw: ImageProfile = profile_images(
            self.df,
            self.rt.store,
            root=self.root,
            sheets_dir=self.rt.ws.work / "sheets",
            class_counts=self.class_counts,
            total_images=self.total,
        )
        self.chart_ids = list(self.raw.chart_ids)
        if len([c for c in self.class_counts if c]) >= 2:
            self._vision()
        return self.raw.quality, None, len(self.df)

    def _vision(self) -> None:
        reporter = self.rt.reporter
        reporter.emit("step", "Vision review of the contact sheets", "one request for all classes")
        try:
            vision_id = vision_review(
                self.rt.store,
                self.raw.sheet_ids,
                tracker=self.rt.tracker,
                route=self.rt.routes["default"],
                api_key=self.rt.api_key,
                on_event=self.rt._on_call,
                client_factory=self.rt.client_factory,
            )
        except Exception as exc:  # the vision review is a bonus; the rest still works
            self.notes.append("The vision review couldn't run, so mislabels weren't checked.")
            reporter.emit("info", "Vision review skipped", str(exc)[:300], "warning")
            return
        if vision_id:
            self.raw.artifact_ids.append(vision_id)
            flagged = self.rt.store.get(vision_id).data["suspected_total"]
            reporter.emit("step", "Vision review done", f"{flagged} suspected mislabels", "done")

    @property
    def guard_df(self) -> pd.DataFrame:
        return self.df

    @property
    def columns(self) -> list[str]:
        return []

    def execute(self, plan: CleaningPlan) -> PlanRun:
        return execute_plan(
            plan,
            self.df,
            self.rt.store,
            catalog=IMAGE_OPS,
            namespace=image_namespace(),
            unit="images",
        )

    def post_checks(self, run: PlanRun) -> list[str]:
        """Reject plans that wipe out most of a class: that usually means a bad threshold."""
        before = self.df.groupby("class").size()
        after = run.df.groupby("class").size().reindex(before.index, fill_value=0)
        return [
            f"The plan removes {1 - after[c] / before[c]:.0%} of class '{c}' ({before[c]} -> "
            f"{after[c]}). Use a gentler threshold or flag instead of dropping."
            for c in before.index
            if c and before[c] >= 5 and after[c] < before[c] * (1 - MAX_CLASS_LOSS)
        ]

    def catalog(self) -> str:
        return catalog_text(IMAGE_OPS)

    def conservative_plan(self) -> CleaningPlan:
        return CleaningPlan(
            summary="Safe-only fallback plan: remove unreadable files, fix orientation, and "
            "convert to RGB. Nothing else is removed.",
            ops=[
                CleaningOp(
                    op="remove_corrupt",
                    rationale="Files that can't be opened.",
                    evidence=[self.raw.artifact_ids[0]],
                ),
                CleaningOp(op="fix_exif_orientation", rationale="Rotate images upright."),
                CleaningOp(op="convert_to_rgb", rationale="Consistent color mode."),
            ],
        )

    def apply(self, plan: CleaningPlan, run: PlanRun) -> tuple[dict[str, str], float, int]:
        out = self.rt.ws.out
        folder = self.rt.ws.work / "cleaned_images"
        export_images(run.df, self.root, folder)
        shutil.copy(folder / "image_manifest.csv", out / "image_manifest.csv")
        archive = shutil.make_archive(str(out / "cleaned_images"), "zip", folder)
        (out / "cleaning_pipeline.py").write_text(
            image_pipeline_script(plan, run, self.rt.ws.input.name), encoding="utf-8"
        )
        kept = run.df.groupby("class").size().to_dict()
        self.clean = profile_images(
            run.df,
            self.rt.store,
            root=self.root,
            sheets_dir=self.rt.ws.work / "sheets",
            class_counts=kept,
            total_images=len(run.df),
            stage="clean",
        )
        self.chart_ids += self.clean.chart_ids[:1]  # class balance after cleaning
        outputs = {
            "cleaned": archive,
            "manifest": str(out / "image_manifest.csv"),
            "pipeline": str(out / "cleaning_pipeline.py"),
        }
        return outputs, self.clean.quality, len(run.df)

    def report_extras(self) -> dict[str, Any]:
        store = self.rt.store
        sheets = [
            {
                "class": store.get(i).data["class"],
                "b64": base64.b64encode(Path(store.get(i).data["path"]).read_bytes()).decode(),
            }
            for i in self.raw.sheet_ids
        ]
        flagged = []
        for category, files in self.raw.categories.items():
            if category == "corrupt" or not files:
                continue
            flagged.append(
                {
                    "category": category.replace("_", " "),
                    "items": [{"path": f, "b64": _thumb_b64(self.root / f)} for f in files[:6]],
                }
            )
        balance = next((a.data for a in store.all("profile") if a.id == "img_balance_001"), None)
        return {"sheets": sheets, "flagged": flagged, "balance": balance}

    def gallery(self) -> list[tuple[str, str]]:
        store = self.rt.store
        items = [
            (store.get(i).data["path"], f"Contact sheet: {store.get(i).data['class']}")
            for i in self.raw.sheet_ids
        ]
        return items


def make_adapter(manifest: FileManifest, rt: JobRuntime) -> Adapter | None:
    return {Modality.TABLE: TableAdapter, Modality.IMAGE: ImageAdapter}.get(
        manifest.dominant, lambda _rt: None
    )(rt)
