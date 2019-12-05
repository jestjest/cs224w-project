#!/usr/bin/python3
#
# CS224W Fall 2019-2020
# @Jason Zheng, Guillaume Nervo, Jestin Ma
#
import datetime as datetime
import numpy as np
import os
import pandas as pd
import pathlib


# Where files listed in DATASETS are located.
DATASET_DIR = '/shared/data'

DATASETS = {
    'bad': [
        'iran_201906_1_tweets_csv_hashed.csv',
    ],
    'benign': [
        'json/democratic_party_timelines',
        'json/republican_party_timelines',
    ]
}

# Where processed datasets will be located.
PROCESSED_DATA_DIR = './datasets/compiled'


# ==============================================================================
# Utilities parsing and processing dataframe values
# ==============================================================================

def pd_float_to_int(float_or_nan):
    """
    @params [float, possibly nan]
    @return [integer, never nan]

    Casts the float as an integer, or 0 if it's nan.
    """
    return 0 if pd.isnull(float_or_nan) else int(float_or_nan)

def pd_str_to_list(str_or_nan):
    """
    @params [a string representing a list or nan]
    @returns [a list]

    Interprets a string or nan as a list. Empty strings = empty lists.
    """
    if pd.notnull(str_or_nan):
        return [] if str_or_nan == '' or str_or_nan == '[]' else str_or_nan[1:-1].split(',')
    else:
        return []

def reformat_datetime(datetime_str, out_format):
    """
    @params [a UTC datetime string with a specific format (see below)]
    @returns: [a date string in the format of out_format]

    Reformats a UTC datetime string returned by the Twitter API for compatibility.
    """

    in_format = "%a %b %d %H:%M:%S %z %Y"
    parsed_datetime = datetime.datetime.strptime(datetime_str, in_format)
    return datetime.datetime.strftime(parsed_datetime, out_format)


# ==============================================================================
# Dataset code
# ==============================================================================

def load_datasets():
    """
    @params: [dataset_grouping (str)]
    @returns: (Pandas Dataframe)

    Reads all csv's from dataset_grouping's input partition in DATASETS, and
    concatenates these to a single pandas dataframe. Returns the dataframe.
    """
    li = []
    for dataset in DATASETS['bad']:
        path = os.path.join(DATASET_DIR, dataset)
        print('Reading data from %s' % path)
        df = pd.read_csv(path)
        df = format_csv_df(df)
        li.append(df)
    return pd.concat(li, axis=0, ignore_index=True)


def load_json():
    """
    @params: []
    @returns: (Pandas Dataframe)

    Reads all json's from dataset_grouping's input partition in DATASETS, and
    concatenates these to a single pandas dataframe. Returns the dataframe.
    """
    li = []
    for dataset in DATASETS['benign']:
        path = os.path.join(DATASET_DIR, dataset)
        print('Reading data from %s' % path)
        df = pd.read_json(path, lines=True)
        df = convert_to_csv_df(df)
        li.append(df)
    return pd.concat(li, axis=0, ignore_index=True)


def format_csv_df(df):
    """
    @params: [df (Pandas Dataframe)]
    @returns: [df (Pandas Dataframe)]

    Selects the relevant fields from csv derived tweet dataframe
    """

    converted_struct = {
        'userid': df['userid'],
        'followers_count': df['follower_count'],
        'following_count': df['following_count'],
        'account_creation_date': df['account_creation_date'],    # Format: YYYY-MM-DD

        'tweet_time': df['tweet_time'],
        'full_text': df['tweet_text'],
        'like_count': df['like_count'].apply(pd_float_to_int),
        'user_mentions': df['user_mentions'].apply(pd_str_to_list),
        'in_reply_to_userid': df['in_reply_to_userid'],     # NaN if not a reply.

        'is_retweeted': df['is_retweet'],
        'retweet_count': df['retweet_count'].apply(pd_float_to_int),
        'retweet_of': df['retweet_userid']      # NaN if not a retweet.
    }
    return pd.DataFrame(converted_struct)


def convert_to_csv_df(df):
    """
    @params: [df (Pandas Dataframe)]
    @returns: [df (Pandas Dataframe)]

    Converts the json structured tweet dataframe to match the CSV bad actors
    dataframe structure
    """
    out_format = "%Y-%m-%d"

    user_metadata = [(
        user.get('id_str'),
        user.get('followers_count'),
        user.get('friends_count'),
        reformat_datetime(user.get('created_at'), "%Y-%m-%d"))
        for user in df['user']]

    unzipped_user_metadata = list(zip(*user_metadata))
    # A list where the Nth item is a list of user mention in the Nth tweet.
    user_mentions = list()
    for entity in df['entities']:
        tweet_mentions = [mention['id'] for mention in entity.get('user_mentions')]
        user_mentions.append(tweet_mentions)


    retweet_status = list()
    for status in df['retweeted_status']:
        if status == status:
            retweet_status.append(str(status['user'].get('id')))
        else:
            retweet_status.append(np.nan)

    converted_struct = {
        'userid': unzipped_user_metadata[0],
        'followers_count': unzipped_user_metadata[1],
        'following_count': unzipped_user_metadata[2],
        'account_creation_date': unzipped_user_metadata[3],

        'tweet_time': df['created_at'].dt.strftime("%Y-%m-%d %H:%M"),
        'full_text': df['full_text'],
        'like_count': df['favorite_count'],
        'user_mentions': user_mentions,
        'in_reply_to_userid': df['in_reply_to_user_id_str'],

        # Existence of this attribute determines whether a tweet is a retweet.
        'is_retweeted': pd.isnull(df['retweeted_status']),
        'retweet_count': df['retweet_count'],
        'retweet_of': retweet_status        # NaN if no retweet
    }

    return pd.DataFrame(converted_struct)


def load_dataset(dataset_name):
    """
    @params: [dataset name in DATASETS]
    @returns: benign_dataset, bad_dataset

    Given a dataset name, loads the corresponding dataset specified in macros.
    """

    print('Reading raw dataset for %s' % dataset_name)
    if dataset_name == 'benign':
        return load_json()

    else:
        return load_datasets()
