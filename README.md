# Optiver - Trading at the Close Predict US

## Project Description
This project was developed as part of the DS 5220 Supervised Machine Learning course in Fall 2023. It focuses on predicting the closing prices of stocks listed on NASDAQ during the Closing Cross auction, a critical event in daily trading that determines official closing prices for securities.

Our team participated in the Kaggle competition "Optiver - Trading at the Close," where we developed a machine learning model to forecast the movement of the Weighted Average Price (WAP) one minute into the future. The project demonstrates the application of advanced supervised learning techniques to real-world financial data, emphasizing the challenges and intricacies of stock market prediction.

Key aspects of the project include:
- Analysis of high-frequency trading data from NASDAQ
- Implementation of various machine learning models, including LightGBM, XGBoost, and Neural Networks
- Extensive feature engineering to capture market dynamics
- Focus on computational efficiency for real-time prediction scenarios
- Evaluation using Mean Absolute Error (MAE) metric

At the time of completing this course project, our team ranked in the top 25% of the Kaggle competition, showcasing the effectiveness of our approach in a competitive, real-world scenario.

This repository contains our codebase, detailed documentation, and a comprehensive report (main.pdf) outlining our methodologies, analyses, and findings.




## Introduction and Objectives
The financial markets are characterized by their complexity and volatility, making accurate predictions of stock prices a challenging yet crucial endeavor. This project focuses on predicting the closing prices of stocks listed on NASDAQ, particularly during the Closing Cross auction, which is a significant event that determines the official closing prices for securities. The primary objective is to develop a robust predictive model capable of forecasting the movement of the Weighted Average Price (WAP) just one minute into the future. This task is not only aimed at achieving a competitive ranking in a Kaggle competition but also at contributing to a deeper understanding of market dynamics during the closing auction, a period marked by heightened trading activity.

For more detailed information and access to the dataset, please visit the Kaggle competition page:
[Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close)

## Significance of the Closing Auction
The Closing Cross auction plays a pivotal role in the trading ecosystem, serving as a convergence point for buyers and sellers. This auction is critical for price formation and market liquidity, as it accounts for nearly 10% of NASDAQ's average daily volume. The auction process is designed to match buy and sell orders at a single price, which is determined based on the supply and demand dynamics at that moment. Understanding the intricacies of this process is essential for developing effective predictive models, as it directly influences the closing prices of stocks. The auction's significance is underscored by its impact on various investment strategies, making it a focal point for traders and analysts alike.

## Data Acquisition and Description
The dataset utilized for this project is sourced from the NASDAQ Closing Cross auction and comprises several key components that provide unique insights into market behavior. The data is provided by Kaggle and includes the following files:

- `[train/test].csv`: The auction data. The test data is delivered by the API.
- `sample_submission`: A valid sample submission, delivered by the API.
- `revealed_targets`: Provides true target values for the entire previous date when `seconds_in_bucket` equals zero.
- `public_timeseries_testing_util.py`: An optional file for running custom offline API tests.
- `example_test_files/`: Data illustrating how the API functions.
- `optiver2023/`: Files that enable the API.

Key features in the dataset include:
- stock_id: A unique identifier for the stock.
- date_id: A unique identifier for the date.
- imbalance_size: The amount unmatched at the current reference price (in USD).
- imbalance_buy_sell_flag: An indicator reflecting the direction of auction imbalance.
- reference_price: The price at which paired shares are maximized.
- matched_size: The amount that can be matched at the current reference price (in USD).
- far_price and near_price: Crossing prices based on different order considerations.
- bid/ask prices and sizes: Information on the most competitive buy/sell levels.
- wap: The weighted average price in the non-auction book.
- seconds_in_bucket: The number of seconds elapsed since the beginning of the day's closing auction.
- target: The 60-second future move in the wap of the stock, less the 60-second future move of the synthetic index (only provided for the train set).

## Statistical and Machine Learning Techniques
The project employs a variety of statistical tools and machine learning algorithms for stock market prediction:
- **Statistical Tools**: Techniques such as ARIMA (Autoregressive Integrated Moving Average) and clustering methods are used for initial data interpretation and to categorize stocks into homogenous groups. This categorization aids in risk diversification and allows for more tailored predictive modeling.
- **Machine Learning Algorithms**: The project leans heavily on advanced machine learning techniques, particularly tree-based models like LightGBM and XGBoost. These models are favored for their ability to handle complex, nonlinear patterns in financial data, making them well-suited for the intricacies of stock price movements. Additionally, Support Vector Machines (SVM) are explored for their efficacy in discerning patterns based on historical data.

## Feature Engineering
A critical aspect of the project is the feature generation process, which involves creating new variables that capture the dynamics of the stock market. Key features include:
- **Price Change Differences**: By calculating differences in price changes across various time windows, the model assesses momentum and volatility, providing insights into the aggressiveness of buyers versus sellers. This feature is essential for understanding market sentiment and predicting price movements.
- **Order Book Details**: Incorporating micro-level details from the order book, such as bid/ask sizes and prices, reflects the competitive landscape of the market. These details are crucial for capturing the immediate supply and demand dynamics that influence stock prices.
- **Macro-Level Market Conditions**: The model also considers broader market indicators, such as trading volumes and global market movements, which may influence stock prices. This holistic approach to feature engineering is essential for enhancing the model's predictive power.

For a more comprehensive understanding of the project and the features we used, please refer to the full report in the `main.pdf` file.
## Model Training and Computational Efficiency
The project emphasizes the importance of computational efficiency, particularly given the time-sensitive nature of stock market predictions. The total execution time for the final Kaggle submission was approximately 5.7 hours, which included:
- **Data Preparation**: Significant time was spent on cleaning and organizing the data to ensure its quality and relevance for modeling. This step is crucial, as the accuracy of predictions is heavily dependent on the quality of the input data.
- **Model Training**: The process of training the LightGBM model involved extensive hyperparameter tuning to optimize performance. This tuning process is critical for achieving the best possible predictive accuracy while maintaining computational efficiency.
- **Submission Process**: Generating predictions and handling data features in compliance with Kaggle's API requirements added to the overall execution time. The model was designed to operate within a total runtime of less than 9 hours, ensuring practical applicability in real-world scenarios where timely decisions are crucial.

## Evaluation Metrics
The performance of the predictive model is evaluated using the Mean Absolute Error (MAE), which measures the average magnitude of errors in predictions without considering their direction. The formula is given by:

MAE = (1/n) * Σ|yi - xi|

where n is the total number of data points, yi is the predicted value, and xi is the observed value. A lower MAE indicates higher accuracy, and the goal is to achieve a MAE significantly lower than that of baseline models. This metric is particularly relevant in financial contexts, where precision in predictions can have substantial implications for trading strategies.

## Results and Learning Outcomes
The project culminated in a successful ranking within the top 25% of the Kaggle competition, demonstrating the model's effectiveness in predicting closing prices. Key learning outcomes include:
- **Adaptability and Learning**: The team transitioned from novices in stock market analytics to developers of a competitive predictive model, showcasing significant growth and adaptability. This journey involved extensive research and experimentation, highlighting the importance of continuous learning in the field of data science.
- **Community Engagement**: Participation in the Kaggle competition discussion board provided valuable insights and fostered a collaborative environment that enhanced understanding of stock market prediction. The community's willingness to share knowledge and experiences played a crucial role in guiding the team's modeling choices and feature engineering strategies.

## Challenges Encountered
The project faced several challenges, including:
- **Understanding Financial Data**: The complexities of financial data, characterized by volatility and numerous influencing factors, required deep exploration of financial theories and market behaviors. This understanding is essential for developing models that can accurately capture the nuances of stock price movements.
- **Feature Engineering**: Developing features that accurately capture market dynamics was both challenging and crucial for model performance. The team had to experiment with various feature combinations to identify those that provided the most predictive power.
- **Model Tuning**: Balancing prediction accuracy with computational efficiency necessitated extensive experimentation with various algorithms and hyperparameters. This process was time-intensive but critical for enhancing the predictive power of the models.

## Conclusions and Future Directions
The project concludes with a reflection on the successful application of machine learning techniques in financial markets, highlighting the balance achieved between computational efficiency and predictive accuracy. The experience gained from this project is seen as a significant step in the team's development in stock market analytics, with implications for future research and practical applications in real-time stock market analysis.

Future directions for this research could include:
- **Exploring Additional Data Sources**: Incorporating alternative data sources, such as social media sentiment or macroeconomic indicators, could enhance the model's predictive capabilities.
- **Advanced Modeling Techniques**: Investigating more sophisticated modeling techniques, such as deep learning or ensemble methods, may yield further improvements in prediction accuracy.
- **Real-Time Implementation**: Developing a framework for real-time prediction and analysis could provide valuable insights for traders and investors, allowing for more informed decision-making in fast-paced market environments.

## Project Structure
|
├── data/
├── models/
│ ├── LightGBM/
│ ├── XGBoost/
│ └── Neural_Network/
├── notebooks/
├── src/
├── README.md
├── requirements.txt
└── .gitignore



## How to Run
1. Clone the repository
2. Navigate to the specific model directory (e.g., `models/LightGBM/`)
3. Run the Jupyter notebooks or Python scripts as needed

Note: The dataset used in this project is not included in this repository due to competition rules. To access the data, please visit the Kaggle competition page:
[Optiver - Trading at the Close](https://www.kaggle.com/competitions/optiver-trading-at-the-close)
You will need to accept the competition rules to download the dataset.

## Contributors
- Raghu Ram Sattanapalle
- Eshan Arora
- Pazin Tarasansombat

## License
This project is licensed under the MIT License - see the LICENSE file for details.


For a more comprehensive understanding of the project, including detailed methodologies, analyses, and results, please refer to the full report in the `main.pdf` file.

Note: At the time of writing the report, our team was ranked in the top 25% of the Kaggle competition. However, we had to discontinue our participation when the semester ended, as we could no longer compete as a group.