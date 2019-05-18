import boto3 as boto3
import pandas as pd
import util as util
from sqlalchemy import create_engine

global engine
# for chennai, where I've taken the database dump and
# renamed cognicity -> riskmap by search replacing in the .sql file.
DATABASE = "riskmap"
engine = create_engine("postgresql://postgres:postgres@localhost:5432/"+ DATABASE)

global client
client = boto3.client('comprehend', region_name='us-east-1')

def get_flood_keys(en=engine):
    start_known_flood = "'2017-11-01 00:00:35.630000-04:00'"
    end = "'2017-11-07 00:00:35.630000-04:00'"
    known_flood_chennai = util.get_flood_pkeys(start_known_flood, end, engine)

def get_all_report_text(en=engine):
    rows = pd.read_sql_query('''
    SELECT pkey, text from riskmap.all_reports
        WHERE text IS NOT null AND LENGTH(text) > 0
        AND text  NOT SIMILAR To '%%(T|t)(E|e)(S|s)(T|t)%%' 
        ORDER BY created_at
    ''', con=en, index_col="pkey")
    return rows

def get_sentiment(report_text):
    # only valid LanguageCode is 'en' or 'es'
    response = client.detect_sentiment(
            Text=report_text,
            LanguageCode='en'
            )
    return response['SentimentScore']

def get_all_sentiments(reportTextDf, hook=None):
    '''
        returns a dictionary of pkey: {text: , SentimentScore: { Neutral ...} }  
    '''
    sentiments = dict()
    for index, row in reportTextDf.iterrows():
        print(row['text'], get_sentiment(row['text']))
        sentiments[index] = {
                'text': row['text'],
                'SentimentScore':get_sentiment(row['text'])
                }
        if hook:
            hook(sentiments)
    return sentiments
