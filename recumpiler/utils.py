#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""
import configparser
import os
import sqlite3

import pandas

from recumpiler.mutators import data_path


def get_default_mutator_config() -> configparser.ConfigParser:
    """Read and return the default recumpiler mutator config"""
    config = configparser.ConfigParser()
    config.read(os.path.join(data_path, "recumpiler.ini"))
    return config


def get_emoji_database():
    emoji_database = sqlite3.connect(":memory:")
    with open(
        os.path.join(
            data_path, "emoji-sentiment-data", "Emoji_Sentiment_Data_v1.0.csv"
        ),
        newline="\n",
        encoding="utf-8",
    ) as csvfile:
        df = pandas.read_csv(csvfile)
        df.to_sql("Emoji_Sentiment_Data", emoji_database, if_exists="fail", index=False)
    return emoji_database
