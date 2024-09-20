# RexNet (Return Expectation Network)

## Project Overview
RexNet is a deep learning model designed to predict stock market returns, aiming to enhance investment strategy efficiency. The model leverages various fundamental, technical, and market data to forecast future stock returns and construct a long-short portfolio based on these predictions. This project focuses on implementing and evaluating a simulated investment strategy.

## Key Features
- **LSTM and CNN-LSTM Models**: Utilizes LSTM and CNN-LSTM architectures to capture temporal patterns in stock data.
- **Comprehensive Data Features**: Combines fundamental data (e.g., BPS, PER, PBR), technical indicators (e.g., closing prices, trading volume), and market indices (e.g., KOSPI index) to optimize prediction performance.
- **Long-Short Portfolio Strategy**: Constructs a portfolio by selecting top and bottom-performing stocks based on predicted returns, aiming to maximize profitability.

## Installation

You can install the required packages with the following commands:

```bash
pip install pykrx
pip install pandas
pip install numpy
pip install torch
pip install scikit-learn
pip install matplotlib
```

## Usage

1. **Data Preparation**: Store all ticker-specific data and market factors in the `/content/drive/My Drive/ticker_data/` directory. Each file should be named in the format `<ticker>_features.csv`.

2. **Model Training**: Run the provided Jupyter Notebook file to train the model. Adjust hyperparameters as needed to optimize model performance.

3. **Portfolio Prediction**: Construct a long-short portfolio based on the predicted returns and visualize the expected cumulative returns to evaluate the strategy's performance.

4. **Visualization & Analysis**: Perform various visualizations using the trained model's predictions to analyze the effectiveness of the investment strategy.

## Data Structure

- **Fundamental Data**: Includes BPS, PER, PBR, EPS, DIV, DPS, and closing prices for each stock ticker.
- **Technical Indicators**: Various technical indicators such as trading volume and moving averages.
- **Market Data**: Overall market indicators, including the KOSPI index.

## Results & Performance

- Provides predicted returns for each stock ticker.
- Constructs a long-short portfolio and visualizes its cumulative returns over the forecast period.
- Evaluates the performance of the simulated investment strategy based on predicted results.

## How to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- This project is based on the latest research in stock market prediction using deep learning and was made possible with easy access to KOSPI data through the `pykrx` library.
- Special thanks to those who provided data and insights necessary for implementing the simulated investment strategy.
