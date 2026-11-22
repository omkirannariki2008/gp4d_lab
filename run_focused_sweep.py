import sys
from gp4d_lab.experiment import ExperimentRunner
from gp4d_lab.focused_experiments import run_all_phases

def main():
    csv_path = "/home/omkiran/kishore/gp4d_lab/data/inclusive_A1__bSat_STU_q0_bin0-1079999_LQQ.csv"
    
    runner = ExperimentRunner(csv_path)
    # 5% training data, 95% validation data
    runner.base_cfg.data.train_fraction = 0.1
    runner.base_cfg.training.device = "auto"
    runner.base_cfg.training.predict_chunk_size = 5000# chunk size to prevent OOM
    
    print(f"Running focused coordinate descent sweep on: {csv_path}")
    print("Kinematic Training Split: 5% train / 95% validation")
    
    # Run targeted 3-phase sweep
    run_all_phases(runner, csv_path, n_epochs=100,verbose=True)

if __name__ == "__main__":
    main()
