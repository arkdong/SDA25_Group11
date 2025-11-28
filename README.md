 # SDA25_Group11

data/		raw data<br/>
results/	figures and tables<br/>
code/		python code used to produce the figures/models

Dataset and helper functions
#### Bitcoin tweets - 16M tweets
- Scrapped from twitters from 2016-01-01 to 2019-03-29, Collecting Tweets containing Bitcoin or BTC
- The columns contain: user, fullname, tweet-id,timestamp, url, likes,replies,retweets, text
- (Link to dataset)[https://www.kaggle.com/datasets/alaix14/bitcoin-tweets-20160101-to-20190329/data]
- Helper function to load dataset:
```python
def load_16M_tweet(columns, csv_path):
    """
    Load the 16M Tweet dataset CSV using a predefined schema.

    Parameters
	    - columns : list[str]
	        List of column names to select. If empty, all columns are returned.
	    - csv_path : str
	        Path to the tweet CSV file based on the current location.

    Returns
	    - pl.DataFrame
	        A Polars DataFrame containing either all the columns or the specified subset.
    """
```
#### Bitcoin tweets - Market Sentiment
- Scrapped from twitters from 2016-01-01 to 2019-03-29, Collecting Tweets containing Bitcoin or BTC with sentiment in positive or negative
- The columns contain: date, text, sentiment (all lowercase)
- (Link to dataset)[https://www.kaggle.com/datasets/gauravduttakiit/bitcoin-tweets-16m-tweets-with-sentiment-tagged/data]
- Helper function to load dataset:
```python
def load_tweet_with_sentiment(columns, csv_path):
	"""
	Load a sentiment-labeled tweet CSV file using a predefined schema.

	Parameters
		- columns : list[str]
			List of column names to select. If empty, all columns are returned.
		- csv_path : str
			Path to the sentiment tweet CSV file (comma-separated).

	Returns
		- pl.DataFrame
			A Polars DataFrame containing either all the columns or the specified subset.
	"""
```
#### Reddit Comments Containing "Bitcoin" 2009 to 2019
- 4M+ Comments from Reddit that contain the word "bitcoin" from 2009 to 2019 collected from Google BigQuery.
- The columns contain: datetime, date, author, subreddit, created_utc, score, controversiality, body
- (Link to dataset)[https://www.kaggle.com/datasets/jerryfanelli/reddit-comments-containing-bitcoin-2009-to-2019]
- Helper function to load dataset:
```python
def load_reddit_comments(columns, csv_path):
	"""
	Load a Reddit comments CSV file using a fixed schema.

	Parameters:
		- columns : list[str]
			List of column names to select. If empty, all columns are returned.
		- csv_path : str
			Path to the Reddit comments CSV file.

	Returns
		- pl.DataFrame
			A Polars DataFrame containing either all columns or the specified subset.
	"""
```


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
