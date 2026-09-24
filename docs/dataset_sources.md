# Dataset sources, provenance and licensing

All facts below were **verified**, not assumed:

* **Local files:** shape, columns and SHA-256 were measured on the supplied files (Phase 1) and are
  enforced by the contracts in `src/config.py`. `scripts/setup_data.py` re-checks them.
* **Kaggle metadata:** owner, licence field, description text and published file size were read from
  Kaggle's public dataset API (`/api/v1/datasets/view/<ref>` and `/api/v1/datasets/list/<ref>`)
  on **2026-09-24**.
* **Not verified:** we did not download the files from Kaggle again to compare checksums. The
  published byte sizes match our local files exactly (see below).

## Decision: raw data is NOT committed to the repository

`data/raw/` is git-ignored. Each team member installs the files locally and verifies them:

```bash
python scripts/setup_data.py --source "<folder containing the three CSV files>"
```

The script copies (never moves) the files, checks each SHA-256 against the contract, marks the copies
read-only and writes `data/processed/provenance.json`. It never overwrites an existing raw file.

Reasons, per dataset, are given below. In short: one licence is informal, one dataset contains personal
names scraped from a third-party platform with an unverified upstream licence, and one file is a
derivative of unknown origin. Keeping all raw data local is the conservative, uniform choice.

---

## 1. Student Performance — regression track (Review 1)

| Field | Value |
|---|---|
| Kaggle page | https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression |
| Owner | Nikhil Narayan (`nikhil7280`) |
| Last updated on Kaggle | 2023-06-29 |
| Kaggle licence field | "Other (specified in description)" |
| Licence text in the description | "Anyone is free to share and use the data" |
| Synthetic? | **Yes.** The description states: the dataset "is synthetic and created for illustrative purposes". |
| Local file | `Student_Performance.csv` |
| SHA-256 | `93793b00d9026d0b4907df0ca9f88b3696747c7496679d35833bb0fbf9fb57cf` |
| Size | 175,071 bytes (matches the Kaggle-published size of `Student_Performance.csv`) |
| Shape | 10,000 rows × 6 columns |
| Target | `Performance Index` |

**Excluded columns:** none. No identifier and no column that records the target was found.

**Licensing note:** the informal licence text appears to allow sharing. We still do not commit the
file, for consistency with the other datasets and because the licence is not a standard one.
The instructor can confirm whether committing it is preferred.

## 2. 10000 Restaurant Reviews — classification track, Part A (Review 1)

| Field | Value |
|---|---|
| Kaggle page | https://www.kaggle.com/datasets/joebeachcapital/restaurant-reviews |
| Owner | Joakim Arvidsson (`joebeachcapital`) |
| Last updated on Kaggle | 2023-09-03 |
| Kaggle licence field | "Database: Open Database, Contents: Database Contents" (Kaggle's label for the ODbL database licence with the DbCL contents licence) |
| Upstream source named on Kaggle | https://github.com/manthanpatel98/Restaurant-Review-Sentiment-Analysis — **the upstream licence was not verified** |
| Synthetic? | Not stated. The content consists of real-looking user reviews with reviewer names. |
| Local file | `Restaurant reviews.csv` |
| SHA-256 | `43a3950890537219def4af1f7a4473dc0e6fa36dafb2df7a07da723c7fb71517` |
| Size | 3,594,924 bytes (matches the Kaggle-published size of `Restaurant reviews.csv`) |
| Shape | 10,000 rows × 8 columns |
| Label | Derived from `Rating` (see `docs/clarifications.md`, C3) |

**Excluded columns and reasons**

| Column | Reason |
|---|---|
| `Rating` | Label source: the sentiment target is derived from it (direct leakage). |
| `Restaurant` | Identifier of the reviewed business; would let models learn restaurant-level bias instead of text sentiment. |
| `Reviewer` | Personal name: an identifier and a privacy concern. Never shown or modelled. |
| `Metadata` | Reviewer activity counts ("N Reviews , M Followers"), not review content. EDA / team feature engineering only. |
| `Time` | Timestamp, not review content. EDA only. |
| `Pictures` | Picture count, not review content. EDA / team feature engineering only. |
| `7514` | Scraping artefact: one non-null value in 10,000 rows. |

**Licensing note:** ODbL has share-alike and attribution conditions, and the upstream licence of the
scraped reviews is unknown. The reviews also contain personal names. For all three reasons the file
is not committed.

## 3. Cafe Sales — clustering track (Review 2, inspection only)

| Field | Value |
|---|---|
| Kaggle page (original) | https://www.kaggle.com/datasets/ahmedmohamed2003/cafe-sales-dirty-data-for-cleaning-training |
| Owner | Ahmed Mohamed (`ahmedmohamed2003`) |
| Last updated on Kaggle | 2025-01-17 |
| Kaggle licence | **CC BY-SA 4.0** (stated in both the licence field and the description) |
| Synthetic? | **Yes.** The description calls it "synthetic data" that is "intentionally dirty". |
| Original Kaggle file | `dirty_cafe_sales.csv`, 550,305 bytes |
| **Supplied local file** | `Cafe_sales_cleaned.csv`, 643,662 bytes — **a cleaned derivative, not the Kaggle file**. Who cleaned it, and how, is **unknown**. |
| SHA-256 (local) | `41e0f4a082be47049d9ebc3a4e5a05e74e9458c1d987c4502c8a26b8c5bb66a6` |
| Shape | 10,000 rows × 8 columns |

**Observed data quality (Phase 1, supplied file):**
* The placeholder values `ERROR` / `UNKNOWN` remain in `Item` (636 rows), `Payment Method` (599),
  `Location` (1,078) and `Transaction Date` (304).
* The numeric columns contain values equal to the column means (2.949984, 3.028463, 8.924352).
  This is consistent with mean imputation.
* `Total Spent = Quantity × Price Per Unit` holds exactly in 85.4 % of rows.
* `Juice` appears with seven different unit prices, although the author's menu lists one price (3).
  One possible explanation is that missing items were filled with the most frequent value. This is
  a hypothesis only and has not been verified.

**Licensing note:** CC BY-SA 4.0 would allow redistribution with attribution and share-alike. However,
the supplied file is a derivative whose author and cleaning steps are unknown, so it is not committed.
For Review 2 the team should either document where the cleaned file came from, or start from the
original Kaggle file and do the cleaning itself.

**Excluded column:** `Transaction ID` (unique identifier).
