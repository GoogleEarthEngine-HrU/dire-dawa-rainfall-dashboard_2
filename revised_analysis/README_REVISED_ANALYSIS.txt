Dire Dawa extreme rainfall: revised reproducible analysis
========================================================

Run from a terminal after installing requirements:
  python extreme_rainfall_analysis.py --input-dir data --output-dir revised_analysis

The original source CSVs are read only. New outputs are written to revised_analysis.
GEV fitting retains the notebook's L-moment method. POT retains the 95th percentile
and 3-day runs declustering. POT return levels now use a Poisson annual exceedance
probability definition so GEV and POT return periods are comparable. POT bootstrap
intervals account for event-count and excess-sampling variability, but remain
conditional on the selected threshold and declustering rule.

The analysis represents daily area-average CHIRPS rainfall over the Dire Dawa
administrative boundary. It does not represent point-gauge or pixel-level extremes.
See VALIDATION_AND_METHODS_REPORT.json for validation and method notes.
