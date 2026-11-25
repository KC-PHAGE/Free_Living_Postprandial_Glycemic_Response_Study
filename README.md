# Analysis Code for Free-Living Postprandial Glycemic Response (PPGR) Study

Reproducible analysis and figure-generation pipeline accompanying the manuscript:

> **Distribution, Inter-individual Variability, and Intra-individual Proportions of Postprandial Glycemic Responses Under Free-Living Conditions in Non-Diabetic Japanese Adults: A Continuous Glucose Monitoring Study**

## Repository Structure

```text
.
├── LICENSE
├── README.md
├── requirements.txt
├── paper_pipeline/
│   ├── __init__.py
│   ├── 01_prepare_paper_data.py          # Quality filtering, metric computation, and cohort aggregation
│   ├── 02_generate_figures_and_tables.py # Tables 1, 2, S1, S2 (.md) and Figures 2, 3, 4, S1, S2 (.pdf/.svg)
│   └── 03_generate_figure5_spectrums.py  # Figure 5 (A–D individual and 2x2 combined panels, .pdf/.svg)
└── utils/
    ├── __init__.py
    ├── cgm_metrics.py                    # 24-h CGM international consensus metrics
    └── stats_utils.py                    # Statistical summaries, Spearman correlation, and FAO/WHO positive iAUC
```

## Requirements

- Python >= 3.10 (manuscript analyses executed with Python 3.14)
- Install dependencies:

```bash
pip install -r requirements.txt
```

## Expected Input Data Schema

Raw participant-level CGM and meal logs are not included in this repository due to participant privacy and data governance restrictions. To run the pipeline on your own compliant dataset, provide three input files:

1. **Demographics (`--demographics`)**: `.csv` or `.xlsx` (reads Worksheet 2 if `.xlsx`) with columns:
   - `glucose_flight_id`: unique participant identifier
   - `gender` (or localized sex column): `female` / `male`
   - `date_of_birth`: `YYYY-MM-DD` (or Excel serial date)
   - `age_group` (optional): fallback missing-age indicator
2. **CGM Records (`--cgm`)**: `.csv` with columns:
   - `glucose_flight_id`: participant identifier
   - `recorded_at`: ISO-8601 timestamp (e.g., `2025-04-25T20:40:00.000+09:00`)
   - `glucose`: sensor glucose concentration (mg/dL)
3. **Meal Logs (`--meals`)**: `.csv` with columns:
   - `meal_id`: unique meal event identifier
   - `glucose_flight_id`: participant identifier
   - `started_at`: ISO-8601 meal start timestamp
   - `category`: meal category (`breakfast`, `lunch`, `dinner`, `snack`)

## Usage

```bash
# 1. Preprocess CGM/meal logs and compute meal- and person-level metrics
python -m paper_pipeline.01_prepare_paper_data \
  --demographics path/to/demographics.csv \
  --cgm path/to/glucose_records.csv \
  --meals path/to/meals.csv \
  --out-dir ./processed_data

# 2. Generate Tables 1, 2, S1, S2 and Figures 2, 3, 4, S1, S2
python -m paper_pipeline.02_generate_figures_and_tables \
  --in-dir ./processed_data \
  --out-dir ./outputs

# 3. Generate Figure 5 (individual panels A–D and 2x2 combined spectrum panel)
python -m paper_pipeline.03_generate_figure5_spectrums \
  --in-dir ./processed_data \
  --out-dir ./outputs
```

Pass `--encoding cp932` to `01_prepare_paper_data` if reading Shift-JIS/CP932 encoded CSV files.
