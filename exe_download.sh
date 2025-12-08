#!/bin/bash

# from root/experiments

instance="157.151.245.37"

experiments=("Large-Cap" "Medium-Cap" "Small-Cap")

for experiment in "${experiments[@]}"; do
    # Download aggregated_statistics
    scp -r "ubuntu@${instance}:/home/ubuntu/visual-reinforcement-fin-decision-making/experiments/${experiment}/aggregated_statistics" "./${experiment}_aggregated_statistics"
    # Download portfolio_factors
    scp -r "ubuntu@${instance}:/home/ubuntu/visual-reinforcement-fin-decision-making/experiments/${experiment}/portfolio_factors" "./${experiment}_portfolio_factors"
done

