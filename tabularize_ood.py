import os
import json
import pandas as pd

# List of baselines and autocalc (folder names in logs/)
baselines = [
    "greedy_replacement_sequencing_logs",
    "cm_replacement_sequencing_logs",
    "random_replacement_sequencing_logs",
    "rnd_sequencing_logs",
    "info_sequencing_logs",
    "lpm_sequencing_logs",
    "none_sequencing_logs",
    "pretrained_baseline_logs",
    "autocalc_logs"   
]

# Protocols to display
protocols = [f"P{i}" for i in range(12)]

# Directory containing logs
logs_dir = "logs"

# Collect results
results = {}
intervention_steps = {}

for baseline in baselines:
    path = os.path.join(logs_dir, baseline, "benchmark_results.json")
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    with open(path, "r") as f:
        data = json.load(f)
    row = []
    for p in protocols:
        if p in data["final_evals"]:
            mean_success = data["final_evals"][p].get("mean_full_integrated_fractional_success", None)
            row.append(mean_success if mean_success is not None else "N/A")
            # Only collect intervention_steps once (from first baseline found)
            if baseline == baselines[0]:
                intervention_steps[p] = data["final_evals"][p].get("total_interventions", "N/A")
        else:
            row.append("N/A")
    results[baseline] = row

# Prepare LaTeX table
latex_header = "Baseline"
for p in protocols:
    latex_header += f" & {p} ({intervention_steps[p]})"
latex_header += " \\\\"

latex_rows = []
for baseline in baselines:
    display_name = baseline.replace("_sequencing_logs", "").replace("_logs", "").replace("_replacement", " (repl)").replace("autocalc", "AutoCaLC").capitalize()
    row = display_name
    for val in results.get(baseline, ["N/A"]*len(protocols)):
        if isinstance(val, float):
            row += f" & {val:.3f}"
        else:
            row += f" & {val}"
    row += " \\\\"
    latex_rows.append(row)

# Print LaTeX table
print("\\begin{tabular}{l" + "c"*len(protocols) + "}")
print(latex_header)
print("\\hline")
for row in latex_rows:
    print(row)
print("\\end{tabular}")