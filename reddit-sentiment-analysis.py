"""
Reddit Sentiment Analyzer
-----------------------------------
- Auto-loads subreddits from 'list of subreddits.csv'
- Uses data.py (us, blacklist, new_words)
- Saves both top tickers and sentiment analysis to CSV in /results folder
"""

import time
import re
import string
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import emoji
import praw
import pandas as pd
import matplotlib.pyplot as plt
import squarify

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import en_core_web_sm

# import ticker data from your data.py
from data import us, blacklist, new_words

# ---------------- CONFIG ----------------
CFG = {
    "subreddit_file": "list of subreddits.csv",
    "results_dir": "results",  # folder where results will be saved
    "post_flairs": {"Daily Discussion", "Weekend Discussion", "Discussion"},
    "good_authors": {"AutoModerator"},
    "ignore_auth_post": {"example"},
    "ignore_auth_comment": {"example"},
    "unique_comment_per_author": True,
    "post_upvote_ratio": 0.70,
    "post_min_ups": 20,
    "comment_min_score": 2,
    "replace_more_limit": 1,
    "picks": 10,
    "picks_analyze": 5,
    "user_agent": "RedditSentimentAnalyzer/1.0",
    "client_id": "",
    "client_secret": "",
    "username": "",
    "password": "",
}

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- GLOBAL NLP RESOURCES ----------------
SPACY = en_core_web_sm.load()
STOP_WORDS = SPACY.Defaults.stop_words
LEMMATIZER = WordNetLemmatizer()
VADER = SentimentIntensityAnalyzer()
VADER.lexicon.update(new_words)
TOKENIZER = RegexpTokenizer(r"\w+|\$[\d\.]+|http\S+")


# ---------------- HELPER FUNCTIONS ----------------
def read_subreddits(file_path: str) -> List[str]:
    """Read subreddit names from CSV file."""
    file = Path(file_path)
    if not file.exists():
        logging.warning(f"File not found: {file}. Using default subreddit 'wallstreetbets'.")
        return ["wallstreetbets"]

    df = pd.read_csv(file)
    col = df.columns[0]
    subs = [str(x).strip() for x in df[col].dropna().unique().tolist() if str(x).strip()]
    logging.info(f"Loaded {len(subs)} subreddits from {file_path}: {subs}")
    return subs


def clean_comment_text(text: str) -> str:
    """Remove emojis, URLs, numbers, and punctuation."""
    no_emoji = emoji.get_emoji_regexp().sub("", text or "")
    no_urls = re.sub(r"http\S+", "", no_emoji)
    no_nums = re.sub(r"\d+", "", no_urls)
    translator = str.maketrans("", "", string.punctuation)
    return no_nums.translate(translator)


def tokenize_and_lemmatize(text: str) -> List[str]:
    tokens = TOKENIZER.tokenize(text.lower())
    filtered = [t for t in tokens if t not in STOP_WORDS and t.upper() not in us]
    return [LEMMATIZER.lemmatize(t) for t in filtered]


# ---------------- DATA EXTRACTION ----------------
def data_extractor(reddit: praw.Reddit, subs: List[str]) -> Tuple[int, int, Dict[str, int], Dict[str, List[str]]]:
    cfg = CFG
    posts_count = comments_count = 0
    tickers_count, ticker_comments, seen_auth_for_ticker = {}, {}, {}

    for sub in subs:
        logging.info(f"Scraping subreddit: r/{sub}")
        subreddit = reddit.subreddit(sub)
        for submission in subreddit.hot(limit=50):
            author_name = getattr(submission.author, "name", None)
            flair = submission.link_flair_text

            if not author_name or author_name in cfg["ignore_auth_post"]:
                continue
            if submission.upvote_ratio < cfg["post_upvote_ratio"] or submission.ups <= cfg["post_min_ups"]:
                continue
            if flair is not None and flair not in cfg["post_flairs"]:
                continue

            posts_count += 1
            submission.comment_sort = "new"

            try:
                submission.comments.replace_more(limit=cfg["replace_more_limit"])
            except Exception as e:
                logging.debug(f"replace_more failed: {e}")

            for comment in submission.comments:
                c_author = getattr(comment.author, "name", None)
                if not c_author or c_author in cfg["ignore_auth_comment"]:
                    continue
                if comment.score <= cfg["comment_min_score"]:
                    continue
                comments_count += 1

                for raw_word in comment.body.split():
                    word = raw_word.replace("$", "")
                    if not word.isupper() or len(word) > 5:
                        continue
                    if word in blacklist or word not in us:
                        continue

                    if cfg["unique_comment_per_author"] and c_author not in cfg["good_authors"]:
                        if c_author in seen_auth_for_ticker.get(word, set()):
                            break

                    tickers_count[word] = tickers_count.get(word, 0) + 1
                    ticker_comments.setdefault(word, []).append(comment.body)
                    seen_auth_for_ticker.setdefault(word, set()).add(c_author)

    return posts_count, comments_count, tickers_count, ticker_comments


# ---------------- SENTIMENT ANALYSIS ----------------
def sentiment_analysis(tickers_sorted: Dict[str, int], ticker_comments: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    top_symbols = list(tickers_sorted.keys())[:CFG["picks_analyze"]]
    results = {}

    for sym in top_symbols:
        comments = ticker_comments.get(sym, [])
        if not comments:
            results[sym] = {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
            continue

        agg = {"neg": 0.0, "neu": 0.0, "pos": 0.0, "compound": 0.0}
        valid_count = 0

        for cmnt in comments:
            cleaned = clean_comment_text(cmnt)
            tokens = tokenize_and_lemmatize(cleaned)
            vs = VADER.polarity_scores(" ".join(tokens)) if tokens else VADER.polarity_scores(cleaned)
            for k in agg:
                agg[k] += vs[k]
            valid_count += 1

        results[sym] = {k: round(agg[k] / max(valid_count, 1), 3) for k in agg}

    return results


# ---------------- SAVE RESULTS ----------------
def save_results(tickers_count: Dict[str, int], scores: Dict[str, Dict[str, float]]):
    results_dir = Path(CFG["results_dir"])
    results_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save ticker counts
    tickers_df = pd.DataFrame(sorted(tickers_count.items(), key=lambda x: x[1], reverse=True), columns=["Ticker", "Mentions"])
    tickers_path = results_dir / f"top_tickers_{timestamp}.csv"
    tickers_df.to_csv(tickers_path, index=False)

    # Save sentiment analysis
    if scores:
        sentiment_df = pd.DataFrame(scores).T.reset_index().rename(columns={"index": "Ticker"})
        sentiment_path = results_dir / f"sentiment_scores_{timestamp}.csv"
        sentiment_df.to_csv(sentiment_path, index=False)
        logging.info(f"Saved sentiment results → {sentiment_path}")

    logging.info(f"Saved top ticker list → {tickers_path}")


# ---------------- VISUALIZATION ----------------
def print_and_visualize(tickers_count: Dict[str, int], scores: Dict[str, Dict[str, float]]):
    if not tickers_count:
        logging.info("No tickers found with the given filters.")
        return

    sorted_tickers = dict(sorted(tickers_count.items(), key=lambda kv: kv[1], reverse=True))
    top_n = list(sorted_tickers.keys())[:CFG["picks"]]
    counts = [sorted_tickers[s] for s in top_n]

    logging.info(f"Top {CFG['picks']} tickers: {dict(list(sorted_tickers.items())[:CFG['picks']])}")

    if scores:
        df = pd.DataFrame(scores).T.rename(
            columns={"neg": "Bearish", "neu": "Neutral", "pos": "Bullish", "compound": "Compound"}
        )
        print("\nSentiment analysis (top picks analyzed):")
        print(df)

    plt.figure(figsize=(10, 6))
    squarify.plot(sizes=counts, label=[f"{s}: {sorted_tickers[s]}" for s in top_n], alpha=0.7)
    plt.axis("off")
    plt.title(f"Top {CFG['picks']} Most Mentioned Tickers")
    plt.show()

    if scores:
        df = df.astype(float)
        df.plot(kind="bar", title=f"Sentiment for Top {CFG['picks_analyze']} Tickers")
        plt.tight_layout()
        plt.show()


# ---------------- MAIN ----------------
def main():
    start = time.time()
    subs = read_subreddits(CFG["subreddit_file"])

    reddit = praw.Reddit(
        user_agent=CFG["user_agent"],
        client_id=CFG["client_id"],
        client_secret=CFG["client_secret"],
        username=CFG["username"],
        password=CFG["password"],
    )

    posts_count, comments_count, tickers_count, ticker_comments = data_extractor(reddit, subs)

    if not tickers_count:
        logging.info("No tickers collected. Check subreddit list or filters.")
        return

    sorted_tickers = dict(sorted(tickers_count.items(), key=lambda kv: kv[1], reverse=True))
    logging.info(f"Analyzed {comments_count} comments from {posts_count} posts.")

    scores = sentiment_analysis(sorted_tickers, ticker_comments)
    print_and_visualize(sorted_tickers, scores)
    save_results(sorted_tickers, scores)

    logging.info(f"Total execution time: {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
