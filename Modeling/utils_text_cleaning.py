"""This module contains utility functions for text cleaning
"""

# built-in modules
import re
import string

# third-party libraries
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def text_cleaning_pd_series(
    column: pd.Series,
    remove_stopwords=True,
    remove_punctuation=True,
    remove_digits=False,
    stemming=False,
    lemmatization=False,
) -> pd.Series:
    """
    Function to clean text in a pandas series with the following steps:
    - remove newlines, tabs, and carriage returns
    - convert to lowercase
    - remove doi and pmid
    - remove punctuation

    Parameters
    ----------
    column : pd.Series
        The column to clean
    remove_stopwords : bool, optional
        Whether to remove stopwords, by default True
    remove_punctuation : bool, optional
        Whether to remove punctuation, by default True
    remove_digits : bool, optional
        Whether to remove digits, by default False
    stemming : bool, optional
        Whether to perform stemming, by default False

    Returns
    -------
    pd.Series
        The cleaned column

    """
    if stemming is True and lemmatization is True:
        raise ValueError("Cannot perform stemming and lemmatization at the same time")

    column = column.str.replace(r"[\n\t\r]", " ", regex=True)
    column = column.str.lower()
    column = column.str.replace(r"(doi:?\s?\d{2}\.\d{4}\/\S+)", "", regex=True)
    column = column.str.replace(r"(\d{2}\.\d{4}\/\S+)", "", regex=True)
    column = column.str.replace(r"(pmid:?\s?\d{8})", "", regex=True)
    column = column.str.replace(r"(\d{8})", "", regex=True)

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        column = column.apply(
            lambda x: " ".join([word for word in x.split() if word not in stop_words])
        )

    if remove_punctuation:
        column = column.str.replace(r"[{}]".format(string.punctuation), "", regex=True)

    if remove_digits:
        column = column.str.replace(r"\d+", "", regex=True)

    column = column.str.replace(r"\s+", " ", regex=True)

    column = column.apply(lambda x: np.nan if x.isspace() or x == "" else x)

    column = column.apply(lambda x: np.nan if len(x.split()) < 10 else x)

    if stemming:
        ps_stemmer = PorterStemmer()
        column = column.apply(lambda x: " ".join([ps_stemmer.stem(word) for word in x.split()]))

    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        column = column.apply(
            lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
        )

    return column


def text_cleaning_string(
    text: str,
    remove_stopwords=True,
    remove_punctuation=True,
    remove_digits=False,
    stemming=False,
    lemmatization=False,
) -> str:
    """Function to clean text in a pandas series with the following steps:
    - remove newlines, tabs, and carriage returns
    - convert to lowercase
    - remove doi and pmid
    - remove punctuation

    Parameters
    ----------
    text : str
        The text to clean
    remove_stopwords : bool, optional
        Whether to remove stopwords, by default True
    remove_punctuation : bool, optional
        Whether to remove punctuation, by default True
    remove_digits : bool, optional
        Whether to remove digits, by default False
    stemming : bool, optional
        Whether to perform stemming, by default False
    lemmatization : bool, optional
        Whether to perform lemmatization, by default False

    Returns
    -------
    str or NaN
        The cleaned text or NaN if the text is empty or has fewer than 10 words
    """

    if stemming is True and lemmatization is True:
        raise ValueError(
            "Both stemming and lemmatization cannot be True at the same time"
        )

    # Remove newlines, tabs, and carriage returns
    text = re.sub(r"[\n\t\r]", " ", text)

    # Convert to lowercase
    text = text.lower()

    # Remove doi and pmid
    text = re.sub(r"doi:?\s?\d{2}\.\d{4}/\S+", "", text)
    text = re.sub(r"(\d{2}\.\d{4}\/\S+)", "", text)
    text = re.sub(r"pmid:?\s?\d{8}", "", text)
    text = re.sub(r"\d{8}", "", text)

    if remove_punctuation:
        text = re.sub(r"[\{}]+".format(re.escape(string.punctuation)), " ", text)

    if remove_digits:
        text = re.sub(r"\d+", "", text)

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        text = " ".join([word for word in text.split() if word not in stop_words])

    # remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Check if the text is empty or consists solely of whitespace after cleaning
    if text.isspace() or text == "":
        return np.nan

    # Check if the text has fewer than 10 words, return NaN for those
    if len(text.split()) < 10:
        return np.nan

    # stemming / lemmatization
    if stemming:
        ps_stemmer = PorterStemmer()
        text = " ".join([ps_stemmer.stem(word) for word in text.split()])

    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        text = " ".join([lemmatizer.lemmatize(word) for word in text.split()])

    return text
