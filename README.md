# SDA25 Group11: Predicting Bitcoin Price Movements Using Public Sentiment

Crypto price prediction and analysis using sentiment: study on whether public sentiment predicts cryptocurrency volatility or price movements.


**RQ1**: Does Twitter sentiment about Bitcoin have predictive power for short-term (next-day) price movements during 2018–2019?
- Hypothesis: Higher Twitter sentiment is associated with higher next-day Bitcoin returns.

**RQ2**: Does Reddit sentiment correlate with Bitcoin volatility in short time intervals?
- Hypothesis: Increasing negative sentiment leads to increased volatility.
- Hypothesis: Increasing in hype-related keywords predicts positive returns. 

**RQ3**: Does Google search activity related to Bitcoin predict short-term market momentum or volatility?
- Hypothesis: Spikes in “Bitcoin” Google search volume are followed by increase in bitcoin volatility within 24 hours

**RQ4**: Does sentiment disagreement affect price stability?
- Hypothesis: Higher sentiment dispersion, i.e. many positive and negative post on the same day, predict unstable Bitcoin price movements

**RQ5**: Can sentiment features like Twitter, Reddit, Google Trends, and Disagreement improve prediction of next-day Bitcoin returns compared to a baseline price-only model 
- Hypothesis: A predictive model including sentiment features achieves significantly better accuracy than a model using only past Bitcoin prices

#### Datasets
For all datasets we limit the timeframe to 1 January 2018 to 1 January 2019, the datasets we are using for this project are:
- Bitcoin tweets - 16M tweets, [Link to dataset](https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329/data)
- Reddit Comments Containing "Bitcoin" 2009 to 2019, [Link to dataset](https://www.kaggle.com/datasets/jerryfanelli/reddit-comments-containing-bitcoin-2009-to-2019)
- Bitcoin Historical Data with open, high, low, close, volume, [Link to dataset](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)
There two helper function:
- `load_data_sentiment(csv_path)`: load the final dataset with sentiment score
- `load_btc_2018_2019(csv_path, part=None)`: load the specific part of the Bitcoin data.

#### Translation and Sentiment Score
To perform sentiment analysis with Bitcoin price, we need to first translate all tweets and reddit messages into English, and then give a sentiment score. Due to the large size of the datasets, **890K Reddit posts** and **1.2M Tweets**, we uses parallelised code that runs on A100 GPU using Google Colab. [Link to the code](https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data)

For translation we relies on the free model **Helsinki-NLP/opus-mt-mul-en**, which limits the input to 512 tokens, each message must be chunked into smaller segments before translation. This significantly slows down the overall process, even when using a GPU. As a result, the translation took approximately **3 hours** for the 890K Reddit messages and around **18 hours** for the 1.2M Tweets.

For the large-scale sentiment analysis on translated Reddit and Twitter datasets we are using the **CardiffNLP Twitter RoBERTa sentiment model**.



## RQ2

`rq2.ipynb` walks through our sentiment–Bitcoin analysis: loading the merged minute-level datasets, summarizing sentiment by time, correlating sentiment with returns/volatility/volume/price, and fitting simple regressions/visualizations to show the (near-zero) relationships.

Precomputed data files are included: `train1_merged`, `train2_merged`, `train3_merged`, `train4_merged`, `validate_merged`, `test_merged`. Each file already merges Reddit sentiment scores with Bitcoin market data at the minute level. These took significant time to build, so best to use them as a starting point rather than re-running the full preprocessing/sentiment pipeline. If you need to replicate from raw sources, you can, but expect long runtimes, otherwise, load the merged files directly for analysis or modeling.

## Git Workflow

##### Starting on a new feature?
1. Update main
	- `git checkout main`: go to main branch
	- `git pull origin main`: update the main branch locally
2. Create your feature branch
	- `git checkout -b yourname/task-name`: this create a new local branch
3. Do your work + commit
	- `git add .` or `git add yourfile`
	- `git commit -m "message`
4. Push your branch
	- `git push -u origin yourname/task-name`
5. Open Pull Request on Github, get review and merge to main
##### Daily Start Routine
1. Make sure you are on your feature branch
	- `git checkout yourname/task-name`
2. Update local main first (always!)
	- `git checkout main`: go to local main
	- `git pull origin main`: update with remote main
3. Sync your local branch with latest main
	- `git checkout yourname/task-name`: go to your local branch
	- `git rebase main`: rebase the feature branch on top of the updated main
4. Continue coding and push to main as previous
	- `git push`
