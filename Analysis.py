import json
from dataclasses import dataclass
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.pipeline import Pipeline
from TwitterAnalysis.TweetsTool import OlderTweets, StreamTweets
from TwitterAnalysis.PredictionPipeline import setup_preprocessing_pipeline


@dataclass(frozen=True)
class Ratio:
    disaster_num: int
    no_disaster_num: int
    ratio: float


@dataclass
class TwitterAnalysis:
    auth_json_file: str
    model_path: str

    def __post_init__(self):
        with open(self.auth_json_file) as f:
            auth_file = json.load(f)

        self.api_key = auth_file["ApiKey"]
        self.api_secret = auth_file["ApiSecret"]
        self.bearer_token = auth_file["BearerToken"]
        self.access_token = auth_file["AccessToken"]
        self.access_token_secret = auth_file["AccessTokenSecret"]

        self.model = load_model(self.model_path)

        self.twitter_stream = StreamTweets(
            api_key=self.api_key,
            api_secret=self.api_secret,
            bearer_token=self.bearer_token,
            access_token=self.access_token,
            access_token_secret=self.access_token_secret,
        )

        self.old_tweets = OlderTweets(bearer_token=self.bearer_token)
        self._twitter_modes = {
            "history": self.old_tweets,
            "stream": self.twitter_stream
        }

    @staticmethod
    def create_pipeline(target_column_name: str, max_length: int, tokenizer_path: str) -> Pipeline:
        """
        :param target_column_name: column name with raw tweets
        :param max_length:
        :param tokenizer_path:
        :return:
        """
        preprocessing_pipeline = setup_preprocessing_pipeline(
            target_column_name=target_column_name,
            max_length=max_length,
            tokenizer_path=tokenizer_path
        )
        return preprocessing_pipeline

    def get_twitter_modes(self) -> list:
        return list(self._twitter_modes.keys())

    def get_raw_tweets_df(self, mode: str, phrase: str,  limit: int, lang="en"):
        """
        :param mode: mode for getting data from twitter (stream or history)
        :param phrase:
        :param limit:
        :param lang:
        :return:
        """

        tweets_df = self._twitter_modes[mode].get_tweets(phrase=phrase, limit=limit, lang=lang)
        return tweets_df

    def make_predictions(self, pipeline: Pipeline, raw_tweets_df: pd.DataFrame):
        raw_tweets = raw_tweets_df["Tweets"].to_list()
        preprocessed_df = pipeline.fit_transform(raw_tweets_df)

        data = preprocessed_df["TextPadded"].to_list()
        data = np.array(data)
        predictions = self.model.predict(data)
        predictions = ["Disaster" if p > 0.5 else "NoDisaster" for p in predictions]

        raw_tweets_df["Predictions"] = predictions
        raw_tweets_df["RawTweets"] = raw_tweets
        return raw_tweets_df[["RawTweets", "Predictions"]]

    @staticmethod
    def get_ratio(predictions_df: pd.DataFrame) -> Ratio:
        """
        :param predictions_df: dataframe with predictions
        :return:
        """
        classes = predictions_df["Predictions"].to_list()
        disaster_num = classes.count("Disaster")
        no_disaster_num = classes.count("NoDisaster")

        if not no_disaster_num:
            ratio = 1
        else:
            ratio = round(disaster_num / no_disaster_num, 2)

        return Ratio(disaster_num=disaster_num, no_disaster_num=no_disaster_num, ratio=ratio)


if __name__ == '__main__':
    phrase = "war at ukraine"

    ta = TwitterAnalysis(auth_json_file="auth.json", model_path="model/twitter_classification_model.h5")
    pipeline = ta.create_pipeline(target_column_name="Tweets", max_length=15, tokenizer_path="model/tokenizer.pkl")
    raw_tweets = ta.get_raw_tweets_df(mode="history", phrase=phrase, limit=100)
    predictions_df = ta.make_predictions(pipeline=pipeline, raw_tweets_df=raw_tweets)

    predictions_df.to_csv("res.csv")

    ratio_obj = ta.get_ratio(predictions_df)
    d_num, nd_num, ratio = ratio_obj.disaster_num, ratio_obj.no_disaster_num, ratio_obj.ratio
    ratio_dict = ratio_obj.__dict__

    classes = list(ratio_dict.keys())[:2]
    values = list(ratio_dict.values())[:2]

    fig = plt.figure(figsize=(10, 5))
    plt.bar(classes, values)
    plt.title(f"Disaster vs no disaster for phrase: {phrase}, ratio: {ratio}")
    plt.show()
