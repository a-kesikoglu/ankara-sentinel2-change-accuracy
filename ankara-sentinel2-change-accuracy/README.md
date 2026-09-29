# ankara-sentinel2-change-accuracy

Code and data for the article **Design-based accuracy and area estimation of Sentinel-2 land cover change in western
Ankara, Türkiye** (Journal of EngineX: Advanced Engineering and Technology).

Land cover maps of 2018 and 2025 are produced from Sentinel-2 spring and summer composites with a Random Forest trained on
stable consensus pixels of Dynamic World and ESA WorldCover. Change is derived by post-classification comparison (PCC) and by
PCC constrained with change vector analysis (CPCC). All products are evaluated with a stratified random sample of 500 pixels
interpreted on very high resolution imagery for both dates, and accuracies and areas are estimated with design-based
estimators (Olofsson et al. 2014; Stehman 2014).

## Repository structure

| Path | Content |
|---|---|
| `config/config.yaml` | All parameters (study area, periods, cloud masking, classes, classifier settings, strata) |
| `config/study_area.geojson` | Study area (WGS 84) |
| `src/` | Analysis code (see workflow below); `estimators.py` implements the stratified estimators |
| `data/samples/training_points.csv` | Training locations (UTM 36N, grid row/column, class) |
| `data/samples/validation_points.csv` | 500 validation pixels with stratum, coordinates and VHR acquisition dates |
| `data/samples/reference_labels.csv` | Reference labels 2018 and 2025 and interpretation confidence (1 low to 3 high) |
| `data/samples/repeat_labels.csv`, `repeat_key.csv` | Blind repeat interpretation of 100 validation pixels |
| `data/samples/validation_sample_full_table.csv` | One row per validation pixel: coordinates, stratum, reference labels, confidence, Wayback release, VHR acquisition date and resolution, labels of the RF (PCC, CPCC) and Dynamic World maps, CVA magnitude, distance to the nearest training location |
| `data/raw/` | Administrative boundaries used in the figures (FAO GAUL 2015) |
| `outputs/maps/` | GeoTIFF, EPSG:32636, 10 m: RF maps (filtered and raw), maximum class probability, PCC and CPCC change codes (from class × 10 + to class), strata, CVA magnitude (× 1000) |
| `outputs/tables/` | Error matrices in sample counts and area proportions, class and change accuracies, area and transition estimates, agreement with Dynamic World and ESA WorldCover, classifier and feature-set experiments, cloud-threshold and reference-subset sensitivity, repeat interpretation agreement, permutation importance, scene lists, Wayback release records |
| `figures/` | Article figures (300 dpi) and the seasonal composites with valid observation counts |

Class codes: 1 water, 2 tree cover, 3 low vegetation, 4 built-up, 5 bare/sparse.
Strata: 1 to 5 stable classes, 6 CPCC change to built-up, 7 other CPCC change, 8 PCC change without CVA support.

## Workflow

Python 3.12, `pip install -r requirements.txt`, and a Google Earth Engine account (set `gee_project` in
`config/config.yaml`). Run the scripts from `src/`:

1. `gee_download.py`: Sentinel-2 composites, observation counts, Dynamic World, WorldCover and GHSL layers to `data/processed/`
2. `training_samples.py`: stable consensus training points
3. `classify.py`: Random Forest maps of 2018 and 2025
4. `change.py`: PCC, CPCC, transition matrices, strata and validation sample
5. `reference_chips.py`: interpretation sheets from the Esri Wayback service (not included in the repository)
6. `evaluate.py`, `compare.py`, `sensitivity.py`, `diagnostics.py`: accuracy, area estimates and robustness
7. `cloud_sensitivity.py`: cloud probability thresholds of 30, 40 and 50%
8. `repeat_chips.py`, `repeat_agreement.py`: repeat interpretation
9. `figures.py`: figures; `results_registry.py` collects every number reported in the article

Steps 1 to 4 require Earth Engine access and regenerate `data/processed/` (about 550 MB, not included), which the later steps also read.
The main estimates of the article can be checked directly from the included maps and samples with

```bash
cd src
python verify_estimates.py
```

which prints the overall accuracies, the accuracy of the PCC change class, the total change area and the net built-up change
with 95% confidence intervals.

## Data sources and licences

The code and the data produced in this study are released under the MIT licence (see `LICENSE`). Third-party inputs keep
their own terms: Sentinel-2 (Copernicus, free and open), Dynamic World V1 (CC BY 4.0), ESA WorldCover v200 (CC BY 4.0),
GHS-BUILT-S R2023A (European Commission, JRC), FAO GAUL 2015. The reference interpretation used Esri World Imagery Wayback
releases 4756 (12 December 2019) and 26334 (5 August 2026), https://livingatlas.arcgis.com/wayback/. Wayback imagery is not
redistributed; Figure 7 reproduces it with attribution (Esri, Maxar, Vantor).

## Citation

Please cite the article (see `CITATION.cff`).

