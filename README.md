# Beijing Air Quality Forecasting

A deep learning project that uses LSTM and GRU neural networks to predict PM2.5 air pollution levels in Beijing based on historical weather and pollution data.

## Project Overview

Air pollution is a major environmental and public health problem in cities around the world. PM2.5 particles are especially harmful because they can go deep into the lungs and cause serious health issues. This project builds a time series forecasting model to predict future PM2.5 levels so that governments and communities can take action early.

The goal is to achieve a Root Mean Squared Error (RMSE) below 4000 on the Kaggle competition leaderboard.

## Dataset

The dataset contains hourly air quality and weather measurements from Beijing, including:
- PM2.5 concentration (target variable)
- Temperature (TEMP)
- Dew Point (DEWP)
- Pressure (PRES)
- Wind Speed (Iws)
- Other weather features

Data source: Kaggle Beijing Air Quality Competition

## Project Structure

```
Beijing-Air-Quality-Forecasting/
|-- final-air-quality-forecasting-draft.ipynb   # Main notebook with all code
|-- README.md                                    # This file
|-- data/                                        # Folder for datasets
|-- outputs/                                     # Folder for model outputs and submissions
```

## Methods Used

### Data Preprocessing
- Time-based interpolation for missing values
- Cyclical encoding for hour and month features (using sine and cosine)
- Feature scaling with StandardScaler
- 24-hour lookback window for sequence creation

### Models
- **Bidirectional LSTM**: Processes sequences in both forward and backward directions
- **GRU (Gated Recurrent Unit)**: A simpler alternative to LSTM with fewer parameters
- **Ensemble Models**: Combines predictions from multiple models for better results

### Training Techniques
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Dropout regularization
- L2 weight regularization
- Batch normalization

## Experiments

The project includes 15 different experiments testing various configurations:

| Experiment | Description |
|------------|-------------|
| Baseline | Stacked BiLSTM (128-64-32 units) |
| Exp 1 | Increased capacity (256-128-64 units) |
| Exp 2 | Simpler model with high dropout |
| Exp 3-5 | Different learning rates and batch sizes |
| Exp 6-9 | Different optimizers (AdamW, RMSprop, SGD) |
| Exp 11 | GRU architecture |
| Exp 12 | Extended 48-hour lookback window |
| Exp 13 | Batch normalization |
| Exp 14 | L2 regularization |
| Exp 15 | Ensemble of multiple models |

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

## How to Reproduce Results

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Beijing-Air-Quality-Forecasting.git
   cd Beijing-Air-Quality-Forecasting
   ```

2. Install the required packages:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn scikit-learn
   ```

3. Download the dataset from Kaggle and place the CSV files in the `data/` folder

4. Open and run the Jupyter notebook:
   ```bash
   jupyter notebook final-air-quality-forecasting-draft.ipynb
   ```

5. Run all cells in order. The notebook will:
   - Load and explore the data
   - Preprocess features and create sequences
   - Train the baseline model
   - Run all 15 experiments
   - Generate submission files in the `outputs/` folder

## Key Findings

1. **Bidirectional LSTMs work well** for air quality forecasting because they capture context from both past and future observations in the lookback window

2. **Regularization is important** to prevent overfitting on this dataset. Dropout, early stopping, and L2 regularization all help

3. **Ensemble methods give better results** by averaging predictions from multiple different models

4. **Longer lookback windows help** the model see multi-day weather patterns that affect pollution levels

5. **GRUs are a good alternative** to LSTMs with faster training and similar performance

## Results

Experiment 8 appears to be the convincing model but experiment 15, which uses ensemble models is a contender

## License

This project is for educational purposes.