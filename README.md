# Visual Reinforcement Learning for Financial Decision-Making

Reinforcement learning system comparing visual (GAF image-based) and numeric agents for stock trading decisions.

## Requirements

- Python 3.x
- Dependencies: `pip install -r requirements.txt`
- Stock data in `dataset/` directory (CSV files)

## Usage

1. **Start the web GUI:**
   ```bash
   python main.py
   ```

2. **Access the interface:**
   - Open `http://127.0.0.1:5000` in your browser

3. **Run experiments:**
   - Select an experiment (Mini, Small-Cap, Medium-Cap, Large-Cap)
   - Click "Run Experiment"
   - Monitor progress via real-time updates

## Experiment Configuration

Experiments are configured in `experiments/{experiment_name}/config.py`:
- Stock tickers
- Training parameters (epochs, learning rate, batch size)
- Cross-validation settings (K-folds, walk-forward windows)
- Date ranges and technical indicators

## Output

- Trained models saved to `experiments/{experiment_name}/{visual|numeric}_models/`
- Results and logs in `experiments/{experiment_name}/experiment-logs/`

