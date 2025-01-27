# Additional Details: Attack Categories

To add predictions of attack categories to a DataFrame, you can use
`analysis.attack_classification.get_df_with_categories()`.
When no argument is provided, the function adds attack category predictions to
the `Lakera/gandalf-rct-attack-categories` dataset.

## Active learning

To run the active learning tool for attack classification, use:

```bash
python -m analysis.attack_classification.active_learning <name of category, e.g. output_obfuscation>
```

The name of the category should match the YAML file under `analysis/attack_classification_labels`.
For instance, `output_obfuscation` reads and modifies `output_obfuscation.yaml`.
