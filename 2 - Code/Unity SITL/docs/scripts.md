# Script Documentation

This document explains the purpose, inputs, and outputs of the executable scripts located in the `scripts/` directory.

---

## 1. `run_all.sh`
**Purpose**: An automated bash script that executes the complete optimization pipeline across all stages.
**Usage**: `./scripts/run_all.sh`
**Details**:
- Calls `run_optimization.py` sequentially for Stages 1A, 1B, 2, and 3.
- Calls `extract_gains.py` at the end to compile the optimal parameters.
- Outputs all SQLite database tracking to `output/optimization.db`.

---

## 2. `run_optimization.py`
**Purpose**: Uses Optuna to tune hyperparameter gains and neural network architectures using mini-batch domain randomization.
**Usage**: 
```bash
python scripts/run_optimization.py --stage <1A|1B|2|3> --num_trials <N> --db <sqlite_url>
```
**Details**:
- **Stage 1A**: Tunes baseline RISE controller (No Wind).
- **Stage 1B**: Tunes baseline RISE controller (With Wind).
- **Stage 2**: Tunes Neural Network architecture (size, learning rate, leakage).
- **Stage 3**: Tunes Super-Twisting baseline.
- **Output**: Writes trials to the specified Optuna SQLite database. Can be interrupted and safely resumed.

---

## 3. `extract_gains.py`
**Purpose**: Reads the Optuna SQLite database, extracts the absolute best hyperparameters from each study, and compiles them into a single YAML file.
**Usage**:
```bash
python scripts/extract_gains.py --db <sqlite_url> --config conf/config.yaml
```
**Details**:
- Automatically merges the best RISE gains with the best Neural Network architecture to form the complete parameter sets for `BEST_NN` and `BEST_INN`.
- **Output**: Generates `conf/best_gains.yaml`.

---

## 4. `evaluate_robustness.py`
**Purpose**: The final statistical validation script. Proves whether the Integrated Neural Network (INN) provides a statistically significant improvement over the baselines.
**Usage**:
```bash
python scripts/evaluate_robustness.py --num_trials 50 --config conf/config.yaml
```
**Details**:
- Loads the optimized gains from `conf/best_gains.yaml`.
- Generates a fixed array of `N` randomized takeoff conditions (varying X, Y, Z).
- Forces all controllers to fly the exact same `N` conditions to create a paired dataset.
- Runs a Shapiro-Wilk test for normality, followed by a Paired t-test or Wilcoxon Signed-Rank test.
- **Output**: Prints statistical conclusions to the terminal and saves raw paired data to `output/robustness_results.csv`.
