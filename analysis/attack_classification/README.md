# Additional Details: Attack Categories

To add predictions of attack categories to a DataFrame, you can use
`analysis.attack_classification.get_df_with_categories()`.
When no argument is provided, the function adds attack category predictions to
the `Lakera/gandalf-rct-attack-categories` dataset.



## Update Dataset used for Categorization


For labeling the attack categories, we work with a subsampled version of the raw data.
If you need to update it, e.g. after renaming some columns in the raw data, run

```bash
python -m analysis.attack_classification.create_gandalf_rct_subsampled
```

This requires an OpenAI key because it also computes text embeddings.

For some parts of the analysis, we also work with a dataset where we select the last attempt
for each (level, user) pair.

```bash
python -m analysis.attack_classification.create_gandalf_rct_attack_categories
```


## Active learning

To run the active learning tool for attack classification, use:

```bash
python -m analysis.attack_classification.active_learning <name of category, e.g. output_obfuscation>
```

The name of the category should match the YAML file under `analysis/attack_classification_labels`.
For instance, `output_obfuscation` reads and modifies `output_obfuscation.yaml`.
