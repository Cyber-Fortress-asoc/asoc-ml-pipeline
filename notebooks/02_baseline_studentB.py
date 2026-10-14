# %% [markdown]
# Baseline model notebook for Student B
#
# This notebook is for baseline modeling and quick comparison across simple classifiers.
# Results here should be reproducible and should eventually be transferred to the training
# module.

# %%
# TODO: load processed features.
# import polars as pl
# df = pl.read_parquet("data/processed/incident_features.parquet")

# %%
# TODO: split the data with the time-aware logic in `asoc_ml.data.splits`.

# %%
# TODO: train one or more baselines and compare macro-F1.

# %%
# TODO: plot confusion matrix for the best model using matplotlib or seaborn.

# %% [markdown]
# ## Baseline workflow
#
# 1. Load processed feature table
# 2. Create train/validation/test splits
# 3. Train DummyClassifier, LogisticRegression, and XGBoost baseline models
# 4. Compare macro-F1 and confusion matrices
# 5. Save the best model and metadata
