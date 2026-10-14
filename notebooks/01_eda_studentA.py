# %% [markdown]
# EDA notebook for Student A
#
# This notebook is for exploratory work only. The goal is to inspect the raw GUIDE-like
# data, understand data quality, and identify leakage risks before feature engineering is
# finalized.

# %%
# TODO: load Parquet from `data/raw` or an external source, then inspect the dataset.
# Example:
# df = pl.read_parquet("data/raw/incident_data.parquet")
# df.head()

# %% [markdown]
# ## 1. Label distribution
#
# - Check the row-level distribution of `IncidentGrade`
# - Check the incident-level distribution after aggregation
# - Verify whether class imbalance should be handled in training

# %%
# TODO: compute class counts and percentages at the row level.

# %% [markdown]
# ## 2. Missing values
#
# - Review nulls in alert-level and incident-level columns
# - Confirm whether missing values are acceptable or require imputation

# %%
# TODO: inspect missing values by column and by incident id.

# %% [markdown]
# ## 3. Timestamp range
#
# - Confirm the date range for the dataset
# - Check whether there are obvious outliers or malformed timestamps

# %%
# TODO: summarize min/max timestamps and time coverage.

# %% [markdown]
# ## 4. Train/test overlap checks
#
# - Ensure no alert or incident appears leakage across temporal splits
# - Confirm the earliest incident time in training is not after the earliest test time

# %%
# TODO: compute overlap checks before training.

# %% [markdown]
# ## 5. Rows per incident
#
# - Check the number of alerts associated with each incident
# - Identify long-tail incidents and whether aggregation is necessary

# %%
# TODO: count the number of rows per IncidentId and summarize the distribution.

# %% [markdown]
# ## 6. Leakage audit
#
# - Review suspicious columns such as `ActionGrouped` and `ActionGranular`
# - Confirm whether they are available at alert time or only after the fact

# %%
# TODO: audit columns from `asoc_ml.config.LEAKAGE_SUSPECTS` and document assumptions.
