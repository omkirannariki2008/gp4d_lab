import json
import logging
from pathlib import Path
from gp4d_lab.config import make_master_default, ExperimentConfig, AxisKernelConfig
from gp4d_lab.experiment import ExperimentRunner

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("run_master_sweep")

def main():
    csv_path = "/home/omkiran/kishore/gp4d_lab/data/inclusive_A1__bSat_STU_q0_bin0-1079999_LQQ.csv"
    
    # 1. Initialize base config
    base_cfg = make_master_default(csv_path)
    base_cfg.training.n_epochs = 50
    base_cfg.training.device = "auto"
    base_cfg.data.train_fraction = 0.10  # 50 train points / 50 test points
    base_cfg.save_results = True
    base_cfg.results_dir = "results/sweep"

    # Define the sweep configurations
    configs = []

    # A. Aggregation Strategy Sweep (6 configs)
    for strategy in ["hard", "poe", "gpoe", "rbcm", "precision", "learned"]:
        cfg = base_cfg.clone()
        cfg.name = f"agg_sweep_{strategy}"
        cfg.aggregation.strategy = strategy
        cfg.aggregation.learn_weights = (strategy == "learned")
        configs.append(cfg)

    # All 7 composite kernels we wish to sweep
    kernel_options = [
        "matern52",
        "spectral_mixture",
        "rbf",
        "sm_x_matern52",
        "gibbs_power_x_matern52",
        "gibbs_sigmoid_x_matern52",
        "matern52_x_sm"
    ]

    # B. Axis-specific Kernel Sweep (4 experts * 7 kernels = 28 configs)
    for axis in ["q2", "w2", "beta", "z"]:
        for k in kernel_options:
            cfg = base_cfg.clone()
            cfg.name = f"kernel_axis_{axis}_{k}"
            cfg.kernels[axis] = AxisKernelConfig(
                axis=axis,
                composite_kernel=k
            )
            configs.append(cfg)

    # C. Global Kernel Sweep (7 configs)
    # Set all experts to the same composite kernel
    for k in kernel_options:
        cfg = base_cfg.clone()
        cfg.name = f"kernel_global_{k}"
        for axis in ["q2", "w2", "beta", "z"]:
            cfg.kernels[axis] = AxisKernelConfig(
                axis=axis,
                composite_kernel=k
            )
        configs.append(cfg)

    # 2. Write the master config.json containing all configs
    master_config_path = Path("master_config.json")
    config_dicts = [cfg.to_dict() for cfg in configs]
    
    log.info(f"Writing {len(configs)} configurations to {master_config_path}")
    with open(master_config_path, "w") as f:
        json.dump(config_dicts, f, indent=2)

    # 3. Read back the configs from master_config.json to ensure correctness
    log.info(f"Reading configs back from {master_config_path}")
    with open(master_config_path) as f:
        loaded_dicts = json.load(f)
    
    loaded_configs = [ExperimentConfig.from_dict(d) for d in loaded_dicts]

    # 4. Execute the sweep using ExperimentRunner
    log.info(f"Starting sweep of {len(loaded_configs)} configurations...")
    runner = ExperimentRunner(csv_path=csv_path)
    results = runner.sweep(loaded_configs)

    # 5. Print and save leaderboard
    df = runner.print_leaderboard(results, primary_metric="mae")
    runner.save_leaderboard(results, "results/sweep/leaderboard.csv")
    log.info("Sweep completed successfully!")

if __name__ == "__main__":
    main()
