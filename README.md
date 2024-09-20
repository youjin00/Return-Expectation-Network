
# RexNet (Return Expectation Network)

## Project Overview
RexNet is a deep learning model designed for predicting future stock returns. It leverages LSTM and CNN-LSTM architectures to capture temporal patterns in financial data and predict future stock performance. The project aims to build a robust forecasting tool that can be used for constructing long-short portfolios.

## Key Features
- **LSTM and CNN-LSTM Models**: Utilizes two model architectures to capture different aspects of the data.
- **Feature Engineering**: Combines fundamental, technical, and market features for each stock ticker.
- **Long-Short Portfolio Strategy**: Generates predictions for the top and bottom performing stocks to construct a long-short portfolio.
- **GAN Integration**: Implements a GAN model for enhancing the feature importance assessment.

## Installation

To run this project, you will need to install the following dependencies:

```bash
pip install pykrx
pip install pandas
pip install numpy
pip install torch
pip install scikit-learn
pip install matplotlib
```

## Usage

1. **Data Preparation**: Ensure that all the ticker data and market factors are stored in the appropriate directory. The expected directory structure is:

    ```
    /content/drive/My Drive/ticker_data/
    ```

    Each file should be named in the format `<ticker>_features.csv`.

2. **Model Training**: Run the Jupyter Notebook to train the model. Adjust hyperparameters and model settings as necessary.

3. **Prediction**: The model will generate predictions for the next 6 months. These predictions are used to construct a long-short portfolio.

4. **Visualization**: The notebook includes code to visualize the cumulative expected returns of the constructed portfolios.

## Data

- **Fundamental Data**: BPS, PER, PBR, EPS, DIV, DPS, and close prices for each ticker.
- **Market Data**: Includes overall market indicators such as the KOSPI index.

## Results

The project outputs the following:

- Predicted returns for each ticker.
- Long-short portfolio performance over the predicted period.
- Visualization of cumulative returns for the portfolio.

## How to Contribute

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- This project is inspired by recent research in the field of stock market prediction using deep learning.
- Special thanks to the creators of the `pykrx` library for providing easy access to KOSPI stock data.
