### RexNet: Return Expectation Network (CNN-LSTM) Implementation

#### Overview / 개요
This project is a customized implementation of a stock price prediction model based on a CNN-LSTM hybrid architecture, inspired by the paper [Factor-GAN: Enhancing Stock Price Prediction and Factor Investment with Generative Adversarial Networks](https://doi.org/10.1371/journal.pone.0306094) by Jiawei Wang and Zhen Chen. The original framework, Factor-GAN, employs GANs to predict stock prices and optimize factor investment strategies. However, this implementation diverges from the original by integrating Convolutional Neural Networks (CNNs) with Long Short-Term Memory (LSTM) networks, rather than GANs, to capture both spatial and temporal dependencies in financial data.

이 프로젝트는 Jiawei Wang와 Zhen Chen의 논문 [Factor-GAN: Enhancing Stock Price Prediction and Factor Investment with Generative Adversarial Networks](https://doi.org/10.1371/journal.pone.0306094)을 기반으로 하여 CNN-LSTM 하이브리드 구조를 활용한 주식 가격 예측 모델을 커스터마이징한 것입니다. 원래의 Factor-GAN 모델은 GAN을 사용하여 주식 가격을 예측하고 팩터 투자 전략을 최적화합니다. 하지만, 이 구현에서는 GAN 대신 CNN과 LSTM 네트워크를 결합하여 금융 데이터의 공간적 및 시간적 종속성을 포착합니다.

#### Key Features / 주요 특징
1. **Hybrid CNN-LSTM Architecture / 하이브리드 CNN-LSTM 아키텍처**
   - Combines CNN for feature extraction and LSTM for sequential learning.
   - CNN을 통한 특징 추출과 LSTM을 이용한 시계열 학습을 결합.
   
2. **Customized Data Pipeline / 맞춤형 데이터 파이프라인**
   - Processes stock market data including fundamental, technical, and market indicators.
   - 주식 시장 데이터를 처리하며, 기본적, 기술적, 시장 지표를 포함.

3. **Performance Evaluation Metrics / 성능 평가 지표**
   - Includes evaluation metrics like Root Mean Square Error (RMSE) and Sharpe Ratio.
   - RMSE, 샤프 비율 등의 평가 지표를 포함.

#### Model Structure / 모델 구조

1. **Convolutional Neural Network (CNN) / 합성곱 신경망**
   - Extracts spatial features from financial time-series data.
   - 금융 시계열 데이터에서 공간적 특징을 추출.
   - Architecture: Three convolutional layers with ReLU activation functions.
   - 아키텍처: ReLU 활성화 함수를 갖춘 세 개의 합성곱 레이어.

2. **Long Short-Term Memory (LSTM) Network / 장단기 메모리 네트워크**
   - Captures temporal dependencies and trends in stock price movements.
   - 주가 움직임의 시간적 종속성 및 추세를 포착.
   - Architecture: Two LSTM layers followed by fully connected layers.
   - 아키텍처: 완전 연결층으로 이어지는 두 개의 LSTM 레이어.

3. **Final Output Layer / 최종 출력층**
   - Single output node predicting the expected return.
   - 예상 수익률을 예측하는 단일 출력 노드.

#### Data / 데이터
- Data source: Uses historical OHLCV (Open, High, Low, Close, Volume) data, fundamental metrics, and technical indicators for top 200 stocks in the KOSPI index.
- 데이터 소스: 코스피 지수 상위 200개 종목의 과거 OHLCV (시가, 고가, 저가, 종가, 거래량) 데이터, 기본적 지표 및 기술적 지표 사용.
- Data preprocessing: Standardization, missing value imputation, and feature engineering are applied.
- 데이터 전처리: 표준화, 결측값 보완 및 특징 공학 적용.

#### Usage / 사용법
1. **Setup / 설정**
   - Clone the repository and install necessary packages.
   - 레포지토리를 복제하고 필요한 패키지를 설치합니다.
     ```bash
     git clone <repository-link>
     cd <repository-directory>
     pip install -r requirements.txt
     ```

2. **Training / 학습**
   - Use the following command to train the model:
   - 모델을 학습시키기 위해 다음 명령어를 사용합니다:
     ```python
     python train.py --config config.yaml
     ```

3. **Evaluation / 평가**
   - Evaluate the trained model using the test dataset:
   - 테스트 데이터셋을 사용하여 학습된 모델을 평가합니다:
     ```python
     python evaluate.py --model model_checkpoint.pth --data test_data.csv
     ```

4. **Prediction / 예측**
   - Use the trained model for future stock return prediction:
   - 학습된 모델을 사용하여 미래의 주식 수익률을 예측합니다:
     ```python
     python predict.py --model model_checkpoint.pth --data predict_data.csv
     ```

#### References / 참고문헌
- Jiawei Wang, Zhen Chen, *Factor-GAN: Enhancing Stock Price Prediction and Factor Investment with Generative Adversarial Networks*, PLoS ONE, 2024.
- Jiawei Wang, Zhen Chen, *Factor-GAN: 주식 가격 예측 및 팩터 투자 최적화를 위한 GAN 활용 연구*, PLoS ONE, 2024.

#### Future Work / 향후 연구
- Exploring the integration of additional macroeconomic indicators.
- 추가적인 거시경제 지표 통합 연구.
- Enhancing model performance with attention mechanisms.
- 어텐션 메커니즘을 사용하여 모델 성능 향상.

This README provides an overview of the modified CNN-LSTM architecture for stock prediction and offers guidance on how to use the provided code. For more detailed information, please refer to the documentation and comments within the code files.

본 README는 주식 예측을 위한 수정된 CNN-LSTM 아키텍처에 대한 개요를 제공하며, 제공된 코드를 사용하는 방법에 대한 안내를 제공합니다. 
자세한 내용은 코드 파일의 문서 및 주석을 참조하세요.
