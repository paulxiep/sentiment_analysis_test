# sentiment_analysis_test

### Disclaimer

- The goal of this repository is not to achieve SOTA or even decent sentiment analysis performance.
  - Rather, the main goal is to build working and reusable components for sentiment analysis task.
- In the google scraping, google search goes by relevance according to google's algorithm and sometimes search results may not be restricted to what the search terms intend. Double-checking of relevance is beyond the scope of this repository.

### Development log

2023-12-28, 10:30 - Created the repository.

2023-12-28, 15:00 - Added google map review scraping module.

2023-12-29, 15:40 - Added basic tokenization module and researched potential deep learning models to fine-tune

2023-12-30, 22:45 - Experimenting with basic LSTM. Downgraded required Python to 3.8 to enable GPU acceleration on Tensorflow. Have yet to re-test the web scraping module.

2023-12-31, 23:15 - Modularized the LSTM module and data preparation module. Devised my own text slicing data augmentation method. Experimented with tf-idf input method.

### Development plan

1. Find some usable data for sentiment analysis in Thai
    - Google map reviews scraping ✓
2. Tokenization / Data Preparation
    - Group reviews by rating and map to 3 levels of sentiment ✓
    - <u>pythainlp</u> library for tokenization ✓
    - Data preprocessing ✓
    - Text slicing Data Augmentation ✓
3. Model
    - LSTM module ✓
4. Evaluation
5. Clean Up