# 📊 Reddit Sentiment Analyzer

A Python-based tool to analyze **stock market sentiment** from Reddit communities such as **r/wallstreetbets**, **r/stocks**, and **r/investing**.  
It automatically fetches trending posts and comments, extracts stock tickers, performs **sentiment analysis**, and visualizes results with clean data charts.

---

## 🚀 Features

- 🔍 Automatically reads subreddit list from `list of subreddits.csv`
- 💬 Extracts **stock tickers** from Reddit posts and comments
- 🧠 Performs **sentiment analysis** using NLTK’s VADER
- 🧹 Cleans, tokenizes, and lemmatizes text with **spaCy**
- 📊 Displays visual charts of top mentioned tickers and sentiment scores
- 💾 Automatically saves results (`top_tickers` + `sentiment_scores`) as timestamped CSV files

---

## 🧩 Project Structure

```
reddit-sentiment-analyzer/
│
├── reddit-sentiment-analysis.py     # Main script (core logic)
├── data.py                          # Ticker list, blacklist, and custom VADER lexicon
├── list of subreddits.csv           # Subreddits to analyze
├── requirements.txt                 # Project dependencies
├── results/                         # Auto-generated results (CSV outputs)
│    ├── top_tickers_YYYYMMDD_HHMMSS.csv
│    └── sentiment_scores_YYYYMMDD_HHMMSS.csv
└── README.md                        # Project documentation
```

---

## ⚙️ Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/TarunSon1/reddit-sentiment-analyzer.git
cd reddit-sentiment-analyzer
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download Required NLTK & spaCy Data
```bash
python -m nltk.downloader vader_lexicon wordnet stopwords
python -m spacy download en_core_web_sm
```

### 4️⃣ Set Reddit API Credentials
You’ll need Reddit developer credentials (free) from:  
🔗 https://www.reddit.com/prefs/apps  

Then edit the config section in **`reddit-sentiment-analysis.py`**:
```python
"client_id": "YOUR_CLIENT_ID",
"client_secret": "YOUR_CLIENT_SECRET",
"username": "YOUR_REDDIT_USERNAME",
"password": "YOUR_REDDIT_PASSWORD",
```

### 5️⃣ Add Subreddit List
In `list of subreddits.csv`, add subreddit names — either in a column or one per line:
```
subreddit
wallstreetbets
stocks
investing
```

---

## ▶️ Usage

Run the main script:
```bash
python reddit-sentiment-analysis.py
```

The script will:
- Load subreddits from your CSV
- Fetch Reddit posts & comments
- Identify mentioned stock tickers
- Perform sentiment analysis
- Display visual charts
- Save results automatically in `/results`

---

## 📈 Output Example

### 🔹 Top 10 Most Mentioned Tickers
| Ticker | Mentions |
|--------|-----------|
| GME | 145 |
| TSLA | 112 |
| NVDA | 96 |

### 🔹 Sentiment Scores
| Ticker | Bearish | Neutral | Bullish | Compound |
|--------|----------|----------|----------|-----------|
| GME | 0.22 | 0.47 | 0.31 | 0.184 |
| TSLA | 0.18 | 0.41 | 0.41 | 0.219 |

---

## 📊 Example Visualizations

- **Treemap** of the most mentioned tickers  
- **Bar Chart** of sentiment scores for top mentioned stocks  

These appear automatically once analysis completes.

---

## 🧰 Technologies Used

| Category | Libraries / Tools |
|-----------|-------------------|
| Reddit API | `praw` |
| Sentiment Analysis | `nltk`, `VADER` |
| NLP Processing | `spaCy`, `WordNetLemmatizer` |
| Data & Visualization | `pandas`, `matplotlib`, `squarify` |
| Text Cleaning | `emoji`, `re`, `string` |
| Logging & Structure | `logging`, `pathlib` |

---

## 🧠 Customization

At the top of `reddit-sentiment-analysis.py`, you can adjust configuration options:
```python
"post_upvote_ratio": 0.70,    # Minimum upvote ratio for posts
"post_min_ups": 20,           # Minimum upvotes for posts
"comment_min_score": 2,       # Minimum upvotes for comments
"picks": 10,                  # Number of top tickers to display
"picks_analyze": 5            # Number of tickers to analyze sentiment for
```

You can also change the number of posts fetched by editing:
```python
for submission in subreddit.hot(limit=50):
```

---

## 💾 Results & Exports

All outputs are saved automatically under:
```
results/
├── top_tickers_YYYYMMDD_HHMMSS.csv
└── sentiment_scores_YYYYMMDD_HHMMSS.csv
```

You can load them later in Excel, Tableau, or Python for further analysis.

---

## 🧾 Example Folder Layout

```
reddit-sentiment-analyzer/
│
├── reddit-sentiment-analysis.py
├── data.py
├── list of subreddits.csv
├── requirements.txt
├── README.md
└── results/
     ├── top_tickers_20251031_153200.csv
     └── sentiment_scores_20251031_153200.csv
```

---

## 📝 License

This project is released under the **MIT License** — free for personal or commercial use.

---

## 💡 Author

**Tarun Soni**  
GitHub: [https://github.com/Tarunson1](https://github.com/Tarunson1)  


---

> ⭐ *If you found this project useful, don’t forget to give it a star on GitHub!*
