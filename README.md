# CS-433 Project 1: predicting coronary heart disease from BRFSS 2015 survey answers

Binary classification (labels -1 / +1) of coronary heart disease or heart
attack (MICHD) from about 320 survey answers per person. Only the Python
standard library and NumPy are used for the model (matplotlib only for the
exploratory plots).

## Reproducing the submission

1. Put the course data files `x_train.csv`, `y_train.csv`, `x_test.csv` in
   `data/dataset/` (the `data/` folder is not tracked by Git).
2. Use Python 3.9 or newer with NumPy (developed with Python 3.11 and numpy 2.4;
   the code only uses basic NumPy functions that also exist in numpy 1.23).
3. Run `python run.py` from the repository root.

`run.py` trains on all training rows and writes `submission.csv` (about 2 minutes
on a laptop, 16 GB of RAM are enough). Training is plain gradient descent from
zero weights without random numbers, so every run writes the same file. The first
run parses the CSV files once and caches them as `.npy` files in `data/cache/`.

## Method

- **Cleaning (`preprocessing.py`, `preprocessing_rules.py`)**: codes such as
  "don't know" and "refused" become missing values, with per-column rules taken
  from the official BRFSS 2015 codebook and a default rule for the other small
  columns; identifier and constant columns are dropped, as are columns with more
  than 95% missing values; missing values are filled with the median and
  missing-value indicator columns are added; columns with many distinct values are
  clipped to the 1st and 99th percentile; all columns are standardized and then
  clipped to [-10, 10]. Everything is learned on the training rows only
  (`fit`) and applied to any rows (`transform`).
- **Model**: logistic regression with an L2 penalty (lambda = 1e-4) and a weight
  of 3 for the positive class, trained by gradient descent (600 steps, step size
  0.1), in `weighted_logistic.py`. A row is predicted +1 when its probability is
  above 0.42.
- **Threshold**: chosen to maximize F1 on out-of-fold predictions, because the
  leaderboard ranks by F1. The price is an accuracy (about 0.87) below that of
  predicting -1 for everyone (0.9117).

## Evaluation

Stratified 5-fold cross-validation with the cleaning fitted inside every fold
(`cross_validation.py`). Out-of-fold F1 of the final model: 0.426 +/- 0.004.
Leaderboard: F1 0.441, accuracy 0.873. Other scripts:

| File | Purpose |
|---|---|
| `implementations.py` | the six required functions |
| `data_loading.py` | CSV loading with a binary cache |
| `leakage_check.py`, `eda.py`, `special_codes_scan.py`, `numeric_scan.py` | data checks and plots |
| `cross_validation.py` | stratified folds, metrics, baseline models |
| `threshold_sweep.py` | F1 / accuracy as a function of the threshold |
| `ridge_cv.py` | ridge regression over a grid of penalties |
| `weighted_cv.py` | class weights and number of gradient steps |
| `make_submission.py` | earlier unweighted submission script |

Class weights and 600 instead of 300 steps changed the F1 by at most 0.004,
which is within the fold standard deviation. A systematic ablation of the
preprocessing switches and a larger hyperparameter search were not done.

## Reference

CDC, Behavioral Risk Factor Surveillance System 2015 codebook (LLCP):
https://www.cdc.gov/brfss/annual_data/2015/pdf/codebook15_llcp.pdf