
# --------------------------- INSTRUCTIONS --------------------------- #

# chmod +x inference.sh
# ./exe_inference.sh

# ------------------------------ SCRIPT ------------------------------ # 

for experiment in "${experiments[@]}"; do

    # --- 1. load the visual and numeric model checkpoints to the mounted storage --- #
    
    cp -r visual-reinforcement-fin-decision-making-storage/experients/'$experiment'/visual_models visual-reinforcement-fin-decision-making/experiments/'$experiment'
    cp -r visual-reinforcement-fin-decision-making-storage/experients/'$experiment'/numeric_models visual-reinforcement-fin-decision-making/experiments/'$experiment'

    echo "Loaded the model checkpoints to storage"

    echo "--- Executing inference for **$experiment** ---"

    # --- 2. execute the inference phase for each experiment subset --- #
    
    python3 -c "
    # import the required modules
    from src.utils.configurations import load_config
    from src.experiments.temporal_cross_validation import TemporalCrossValidation

    # load the experiment config file
    experiment_name = '$experiment'
    config = load_config(experiment_name)

    # execute the experiment in inference mode; results will save to experiment directory
    experiment = TemporalCrossValidation(experiment_name, config)
    experiment.exe_experiment('inference')
    "
    
    echo " Completed **$experiment** experiment"
    
done

echo "All training experiments completed and saved"