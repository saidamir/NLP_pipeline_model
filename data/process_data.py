import sys
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(messages_filepath, categories_filepath):
# load messages dataset
	messages = pd.read_csv('messages.csv')
	
	categories = pd.read_csv('categories.csv')
	df = pd.merge(messages, categories, how = 'inner',on ='id'  )
	return df

def clean_data(df):
"""create a dataframe of the individual category columns"""
	categories = df['categories'].str.split(pat = ';', expand = True)
	categories.columns = categories.iloc[1]
	categories = categories.applymap(lambda x: int(x[-1]))
	categories.columns = categories.columns.map(lambda x: x[:-2]).rename(None)
	df.drop(columns='categories', inplace=True)
	df = pd.concat([df,categories],axis=1)
	#drop duplicated row
	df.drop_duplicates(inplace=True)
	print ("Duplicates: ", df.duplicated().sum())
	return df

def save_data(df, database_filename):
	"""Saves dataframe to database"""
	name = 'sqlite:///' + database_filename
	engine = create_engine(name)
	df.to_sql('InsertTableName', engine, index=False, if_exists='replace')

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()




