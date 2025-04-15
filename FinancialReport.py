import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
hg_embeddings = HuggingFaceEmbeddings()

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen

import certifi
import json
import pandas as pd


def get_jsonparsed_data(url, api_key, exchange):
  if exchange == "NSE":
    url = f"https://financialmodelingprep.com/api/v3/search?query={ticker}&exchange=NSE&apikey={api_key}"
  else:
    url = f"https://financialmodelingprep.com/api/v3/quote/{ticker}?apikey={api_key}"
  response = urlopen(url, cafile=certifi.where())
  data = response.read().decode("utf-8")
  return json.loads(data)

api_key="Enter Your API Key"
ticker = "MSFT"
exchange = "US"
eco_ind = pd.DataFrame(get_jsonparsed_data(ticker, api_key,exchange))
eco_ind

def preprocess_economic_data(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['earningsAnnouncement'] = pd.to_datetime(df['earningsAnnouncement'])
    return df

preprocessed_economic_df = preprocess_economic_data(eco_ind)
preprocessed_economic_df

preprocessed_economic_df.to_csv("Enter Your File Location")
