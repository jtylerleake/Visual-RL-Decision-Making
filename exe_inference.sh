
experiments=("Large-Cap" "Medium-Cap" "Small-Cap")

for experiment in "${experiments[@]}"; do

    cd ~
    cp -v visual-reinforcement-fin-decision-making-storage/experiments/"${experiment}"/visual_models visual-reinforcement-fin-decision-making/experiments/"${experiment}"
    cp -v visual-reinforcement-fin-decision-making-storage/experiments/"${experiment}"/numeric_models visual-reinforcement-fin-decision-making/experiments/"${experiment}"

done 

cd visual-reinforcement-fin-decision-making


for experiment in "${experiments[@]}"; do

    echo "--- Executing inference for **$experiment** ---"
    
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

echo "All experiments completed in inference mode"
echo "Exiting..."