from time import sleep
from dataclasses import dataclass
from collections import deque
import tweepy
from langdetect import detect, lang_detect_exception
import pandas as pd


@dataclass
class OlderTweets:
    bearer_token: str

    def __post_init__(self):
        self.client = tweepy.Client(bearer_token=self.bearer_token)

    def get_tweets(self, phrase: str, limit=100, lang="en") -> pd.DataFrame:
        query = f"{phrase} -is:retweet lang:{lang}"
        tweets = tweepy.Paginator(self.client.search_recent_tweets, query=query, max_results=100).flatten(limit=limit)
        tweets_list = [str(tweet).replace("\n", " ") for tweet in tweets]

        return pd.DataFrame({
                "Tweets": tweets_list
            })


class TwitterStreaming(tweepy.StreamingClient):
    def __init__(self, lang="en", maxlen=100, **kwargs):
        self.maxlen = maxlen
        self.lang = lang
        self.tweets_deq = deque(maxlen=self.maxlen)
        super(TwitterStreaming, self).__init__(**kwargs)

    def on_connect(self) -> None:
        print("Connected")

    def on_tweet(self, tweet) -> None:
        if len(self.tweets_deq) == self.maxlen:
            self.disconnect()

        elif tweet.referenced_tweets:
            try:
                tweet_lang = detect(tweet.text)
                if self.lang == tweet_lang:
                    print(tweet.text)
                    self.tweets_deq.append(tweet.text.replace("\n", " "))
                    sleep(0.3)
            except lang_detect_exception.LangDetectException:
                pass


@dataclass
class StreamTweets:
    api_key: str
    api_secret: str
    bearer_token: str
    access_token: str
    access_token_secret: str

    def __post_init__(self):
        self.client = tweepy.Client(self.bearer_token, self.api_key, self.api_secret, self.access_token,
                                    self.access_token_secret)
        auth = tweepy.OAuth1UserHandler(self.api_key, self.api_secret, self.access_token, self.access_token_secret)
        self.api = tweepy.API(auth)

    def get_tweets(self, limit: int, phrase: str, lang="en") -> pd.DataFrame:
        stream = TwitterStreaming(bearer_token=self.bearer_token, lang=lang, maxlen=limit)

        rules = stream.get_rules()
        if rules.data:
            rules_ids = [rule.id for rule in rules.data]
            stream.delete_rules(rules_ids)

        stream.add_rules(tweepy.StreamRule(phrase))


        # Starting stream
        stream.filter(tweet_fields=["referenced_tweets"])

        return pd.DataFrame({
            "Tweets": stream.tweets_deq
        })
