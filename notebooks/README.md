# Notebook rules

- One notebook = one owner.
- Use names like `NN_topic_owner.py` such as `01_eda_studentA.py`.
- Run `nbstripout --install` before committing notebook outputs.
- Use Jupytext paired notebooks (`.py` with `# %%` cells) and keep the source in version control.
- Move reusable code into `src/asoc_ml/` and keep notebooks focused on exploration.
- Treat notebooks as analysis artifacts, not production code.
