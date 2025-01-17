# Gandalf the Red: Adaptive Security for LLMs

This code repository contains the code to reproduce the experiments in the paper [Gandalf the Red: Adaptive Security for Large Language Models](https://arxiv.org/abs/2501.07927).


## Usage

It's recommended to work inside of a [virtualenv](https://virtualenv.pypa.io/en/latest/) or similar (Conda, Poetry) to avoid dependency conflicts.
Before running any of these commands, install dependencies using

```
pip install -r requirements.txt
```

## Reproduce All Plots and Tables

To reproduce all plots and tables from the paper, run:

```bash
python create_all_paper_plots_and_tables.py
```


## Code Structure

The code is structured as follows:

```
analysis
│
├── adaptive_defenses                   # Adaptive defenses experiments
│   └── adaptive_defenses.py
│
├── attack_classification               # Attack classification experiments from Appendix
│   ├── active_learning.py
│   ├── active_learning_data.py
│   ├── create_gandalf_rct_attack_categories.py
│   ├── create_gandalf_rct_subsampled.py
│   ├── labels
│   ├── plots.py
│   ├── predictions.py
│   └── sample_selection.py
│
├── defense_in_depth                    # Defense in depth experiments
│   ├── optimal_aggregation.py
│   └── venn_diagram.py
│
├── supporting_analyses                 # Supporting analyses from Appendix
│   ├── basic_statistics.py
│   ├── false_positives.py
│   ├── level_difficulty.py
│   └── session_length.py
│
├── utility_sensitivity                 # Sensitivity of utility to data and metric experiments
│   ├── sensitivity_to_data.py
│   └── sensitivity_to_metric.py
│
├── embedding_utils.py                  # Auxiliary functions for text embeddings
│
├── utils.py                            # Auxiliary functions used by several scripts
│
create_all_paper_plots_and_tables.py    # Script to reproduce all plots and tables
│
data.py                                 # Script with auxiliary functions to load datasets
```



