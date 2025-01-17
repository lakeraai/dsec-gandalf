# Experiments from Appendix:
## Supporting Analyses of Gandalf-RCT
from analysis.supporting_analyses.session_lengths import main as session_lengths
from analysis.supporting_analyses.basic_statistics import main as basic_statistics
from analysis.supporting_analyses.level_difficulty_and_afr import main as level_difficulty_and_afr

print("Creating session length plots...")
session_lengths()
print("Creating basic statistics tables...")
basic_statistics()
print("Creating percentage of players and AFR tables...")
level_difficulty_and_afr()


# Experiments from Appendix:
## Data collection of BasicUser and BorderlineUser
from analysis.supporting_analyses.false_positives import main as false_positive_rates

print("Creating FPR and accidental reveal tables...")
false_positive_rates()


# Experiments from Appendix:
## Attack Categorization
from analysis.attack_classification.plots import save_attack_categories_plots

print("Creating attack categories plots...")
save_attack_categories_plots()

# Experiments from Section:
## Sensitivity of Utility to Data and Metrics
from analysis.utility_sensitivity.sensitivity_to_data import main as sensitivity_to_data
from analysis.utility_sensitivity.sensitivity_to_metric import main as sensitivity_to_metric

print("Creating plot to assess sensitivity to data...")
sensitivity_to_data()
print("Creating boxplots to assess output degradation...")
sensitivity_to_metric()


# Experiments from Section:
## Defense-in-Depth
from analysis.defense_in_depth.venn_diagram import main as venn_diagram
from analysis.defense_in_depth.optimal_aggregation import main as optimal_aggregation

print("Creating defense-in-depth venn diagrams...")
venn_diagram()
print("Creating optimal aggregations...")
optimal_aggregation()


# Experiments from Section:
## Adaptive Defenses
from analysis.adaptive_defenses.adaptive_defense import main as block_threshold_vs_utility

print("Creating block threshold vs utility plots...")
block_threshold_vs_utility()
