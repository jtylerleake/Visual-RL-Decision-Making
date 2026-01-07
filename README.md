# 🎯 Visual Reinforcement Learning for Financial Decision-Making

<div align="center">

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Flask](https://img.shields.io/badge/flask-2.0+-blue.svg)
![PyTorch](https://img.shields.io/badge/pytorch-2.5+-orange.svg)

**A cutting-edge reinforcement learning system that compares visual (GAF image-based) and numeric agents for intelligent stock trading decisions.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Experiments](#-experiments) • [License](#-license)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Experiments](#-experiments)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Output & Results](#-output--results)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project implements a sophisticated reinforcement learning framework for financial decision-making that leverages two distinct approaches:

- **Visual Agent**: Uses Gramian Angular Field (GAF) transformations to convert time-series stock data into images, enabling convolutional neural networks to learn trading patterns visually
- **Numeric Agent**: Traditional approach using raw numerical features and technical indicators

The system provides a comprehensive web-based interface for running experiments, comparing model performance, and analyzing trading strategies across different market cap categories (Mini, Small-Cap, Medium-Cap, Large-Cap).

---

## ✨ Features

- 🖼️ **Visual Learning**: GAF image-based representation of financial time-series data
- 📊 **Dual Agent System**: Compare visual vs. numeric reinforcement learning approaches
- 🌐 **Web Interface**: Intuitive Flask-based GUI for experiment management
- 📈 **Multiple Market Caps**: Pre-configured experiments for different stock categories
- 🔄 **Temporal Cross-Validation**: Robust K-fold and walk-forward validation strategies
- 🎛️ **Hyperparameter Tuning**: Automated optimization using Optuna
- 📉 **Real-time Monitoring**: Live progress updates during training
- 📋 **Comprehensive Results**: Detailed statistics, figures, and LaTeX tables
- 🧪 **Stable Baselines3**: Industry-standard RL algorithms integration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (Flask)                     │
│              Real-time Updates via SocketIO                   │
└───────────────────────┬─────────────────────────────────────┘
                        │
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐            ┌─────────▼──────────┐
│  Visual Agent  │            │  Numeric Agent     │
│  (GAF Images)  │            │  (Raw Features)    │
│  CNN-based     │            │  MLP-based         │
└───────┬────────┘            └─────────┬──────────┘
        │                               │
        └───────────────┬───────────────┘
                        │
            ┌───────────▼───────────┐
            │  Trading Environment  │
            │  (Gymnasium/AnyTrading)│
            └───────────────────────┘
```

### Key Components

- **GUI Layer**: Flask web application with real-time experiment monitoring
- **Experiment Runner**: Orchestrates training, validation, and evaluation
- **Model Architectures**: CNN for visual, MLP for numeric representations
- **Data Pipeline**: Stock data processing, GAF transformation, feature engineering
- **Validation Framework**: Temporal cross-validation with K-folds and walk-forward windows
- **Hyperparameter Optimization**: Optuna-based automated tuning

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd visual-reinforcement-fin-decision-making
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare stock data:**
   - Ensure CSV files are in the `dataset/` directory
   - Each file should follow the format: `{TICKER}_data.csv`
   - Required columns: Date, Open, High, Low, Close, Volume

---

## 💻 Usage

### Quick Start

1. **Launch the web interface:**
   ```bash
   python main.py
   ```

2. **Access the dashboard:**
   - Open your browser and navigate to `http://127.0.0.1:5000`
   - You'll see the experiment launcher interface

3. **Run an experiment:**
   - Select an experiment type (Mini, Small-Cap, Medium-Cap, or Large-Cap)
   - Click "Run Experiment"
   - Monitor real-time progress through the web interface

### Command Line Usage

You can also run experiments via shell scripts:

```bash
# Setup and training
bash shell/exe_setup.sh
bash shell/exe_training.sh

# Hyperparameter tuning
bash shell/exe_tuning.sh

# Inference and evaluation
bash shell/exe_inference.sh
```

### Generate Figures and Tables

```bash
python experiments/exe_figures.py
```

This will generate:
- Normalized test performance plots
- LaTeX-formatted result tables
- Cumulative return visualizations

---

## 🧪 Experiments

The project includes pre-configured experiments for different market segments:

| Experiment | Description | Tickers |
|------------|-------------|---------|
| **Mini** | Small-scale testing | Limited set |
| **Small-Cap** | Small capitalization stocks | ~20-30 stocks |
| **Medium-Cap** | Mid-capitalization stocks | ~30-50 stocks |
| **Large-Cap** | Large capitalization stocks | 100+ stocks |

Each experiment includes:
- Pre-configured stock tickers
- Optimized hyperparameters (saved in JSON format)
- Trained model checkpoints
- Comprehensive evaluation results

---

## 📁 Project Structure

```
visual-reinforcement-fin-decision-making/
│
├── dataset/                    # Stock data CSV files
├── experiments/                # Experiment configurations and results
│   ├── Mini/
│   ├── Small-Cap/
│   ├── Medium-Cap/
│   └── Large-Cap/
│       ├── config.py          # Experiment configuration
│       ├── figures/            # Generated plots
│       ├── tables/             # LaTeX result tables
│       ├── visual_models/      # Trained visual agent models
│       ├── numeric_models/     # Trained numeric agent models
│       └── experiment-logs/    # Training logs and metrics
│
├── src/                        # Core source code
│   ├── models/                 # Agent model architectures
│   ├── pipeline/               # Data processing pipeline
│   ├── experiments/            # Experiment execution logic
│   └── utils/                  # Utility functions
│
├── gui/                        # Web interface
│   ├── experiment_launcher.py  # Flask app entry point
│   ├── experiment_runner.py    # Experiment execution
│   ├── static/                 # CSS and JavaScript
│   └── templates/              # HTML templates
│
├── shell/                      # Shell scripts for automation
├── common/                     # Shared modules
├── legacy/                     # Legacy code and utilities
│
├── main.py                     # Application entry point
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Configuration

Experiments are configured in `experiments/{experiment_name}/config.py`:

### Key Configuration Options

- **Stock Tickers**: List of stock symbols to include
- **Training Parameters**:
  - Number of epochs
  - Learning rate
  - Batch size
  - Discount factor (gamma)
- **Cross-Validation**:
  - K-fold settings
  - Walk-forward window sizes
  - Train/test split ratios
- **Data Settings**:
  - Date ranges
  - Technical indicators
  - Feature engineering options
- **Model Architecture**:
  - Network layer sizes
  - Activation functions
  - Regularization parameters

Example configuration structure:
```python
TICKERS = ['AAPL', 'MSFT', 'GOOGL', ...]
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 64
K_FOLDS = 5
WALK_FORWARD_WINDOWS = [30, 60, 90]
```

---

## 📊 Output & Results

After running experiments, results are organized as follows:

### Model Checkpoints
- **Location**: `experiments/{experiment_name}/{visual|numeric}_models/`
- **Format**: Saved model weights and optimizer states
- **Usage**: Load for inference or continued training

### Evaluation Metrics
- **Location**: `experiments/{experiment_name}/experiment-logs/`
- **Contents**:
  - Training/validation loss curves
  - Portfolio performance metrics
  - Cumulative returns
  - Sharpe ratios
  - Maximum drawdowns

### Visualizations
- **Location**: `experiments/{experiment_name}/figures/`
- **Types**:
  - Normalized test performance comparisons
  - Best model overlay plots
  - Portfolio factor analysis

### Result Tables
- **Location**: `experiments/{experiment_name}/tables/`
- **Format**: LaTeX-compatible tables
- **Metrics**: Cumulative returns (fold-wise and window-wise)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting:
   ```bash
   pytest
   black .
   flake8 .
   ```
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built with [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for reinforcement learning
- Uses [Gymnasium](https://gymnasium.farama.org/) and [gym-anytrading](https://github.com/AminHP/gym-anytrading) for trading environments
- [Optuna](https://optuna.org/) for hyperparameter optimization
- [PyTS](https://pyts.readthedocs.io/) for time-series transformations

---

<div align="center">

**Made with ❤️ for quantitative finance and reinforcement learning**

[Report Bug](https://github.com/yourusername/visual-reinforcement-fin-decision-making/issues) • [Request Feature](https://github.com/yourusername/visual-reinforcement-fin-decision-making/issues) • [Documentation](#-usage)

</div>
