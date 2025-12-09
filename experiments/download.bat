@echo off
REM navigate to local project ./experiments

set INSTANCE=129.153.47.117
set SSH_KEY=C:\Users\Jtyler\.ssh\id_rsa

REM Loop through experiments
for %%e in ("Large-Cap" "Medium-Cap" "Small-Cap") do (
    set "EXPERIMENT=%%~e"
    
    REM Download aggregated_statistics
    scp -i "%SSH_KEY%" -r "ubuntu@%INSTANCE%:/home/ubuntu/visual-reinforcement-fin-decision-making/experiments/%%~e/aggregated_statistics" ".\%%~e\aggregated_statistics"
    
    REM Download portfolio_factors
    scp -i "%SSH_KEY%" -r "ubuntu@%INSTANCE%:/home/ubuntu/visual-reinforcement-fin-decision-making/experiments/%%~e/portfolio_factors" ".\%%~e\portfolio_factors"
)

echo Download complete!
