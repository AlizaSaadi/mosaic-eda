"""Generate examples/datasets/messy_sales.csv, a deliberately messy demo table.

Planted problems (the evaluation set checks that MOSAIC finds them):
- a title row above the header
- mixed date formats in order_date
- currency strings in unit_price ("$1,204.50"), percentages in discount ("10%")
- hidden missing values ("N/A", "-", "?") in quantity and region
- case/spelling variants in region ("north", "North", "NORTH ")
- about 3% exact duplicate rows
- extreme outliers in revenue
- refund_amount leaks the churned target (non-zero only when churned = yes)
- a constant column (currency) and an empty column (notes)
"""

from __future__ import annotations

import csv
import random
from datetime import date, timedelta
from pathlib import Path

OUT = Path(__file__).parent / "datasets" / "messy_sales.csv"


def main(rows: int = 400, seed: int = 11) -> Path:
    rng = random.Random(seed)
    regions = ["North", "south", "East", "WEST", "north", "North ", "N/A", "East"]
    products = ["Widget", "Gadget", "Doohickey", "Gizmo"]
    base = [12.5, 48.0, 150.0, 1204.5]
    start = date(2025, 1, 1)
    header = [
        "order_id",
        "order_date",
        "region",
        "product",
        "unit_price",
        "quantity",
        "discount",
        "revenue",
        "customer_email",
        "churned",
        "refund_amount",
        "currency",
        "notes",
    ]
    records = []
    for i in range(rows):
        p = rng.randrange(len(products))
        qty = rng.randint(1, 20)
        disc = rng.choice([0, 5, 10, 15, 20])
        price = base[p] * rng.uniform(0.9, 1.1)
        revenue = price * qty * (1 - disc / 100)
        if rng.random() < 0.01:
            revenue *= 40  # planted outliers
        day = start + timedelta(days=rng.randint(0, 364))
        fmt = rng.choice(["%Y-%m-%d", "%m/%d/%Y", "%d %b %Y"])
        churned = "yes" if rng.random() < 0.22 else "no"
        refund = f"{revenue * rng.uniform(0.2, 1):.2f}" if churned == "yes" else "0"
        records.append(
            [
                f"{10000 + i}",
                day.strftime(fmt),
                rng.choice(regions),
                products[p],
                f"${price:,.2f}",
                rng.choice([str(qty)] * 12 + ["N/A", "-", "?"]),
                f"{disc}%",
                f"{revenue:.2f}",
                f"user{rng.randint(1, 300)}@example.com",
                churned,
                refund,
                "USD",
                "",
            ]
        )
    for _ in range(int(rows * 0.03)):
        records.insert(rng.randrange(len(records)), list(rng.choice(records)))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Q3 sales export - generated 2026-09-25", "", "", ""])
        writer.writerow(header)
        writer.writerows(records)
    return OUT


if __name__ == "__main__":
    print(main())
