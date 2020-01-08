# NLP_pipeline_model
## Udacity Project: Disaster Response Pipeline 
In this project I will create a model which will classify disaster response messages using machine learning algorithms.

### Process:
The following folders and processes are used:
Data
process_data.py: reads the data, cleans and uploads it to a SQL database. Basic usage is python process_data.py MESSAGES_DATA CATEGORIES_DATA NAME_FOR_DATABASE
Datasets are disaster_categories.csv and disaster_messages.csv
DisasterResponse.db: resulting database from transformed and cleaned data.

Models
train_classifier.py: has the code to load data, transform it using NLP processing, then run a machine learning model using GridSearchCV and train it. Basic usage is python train_classifier.py DATABASE_DIRECTORY SAVENAME_FOR_MODEL

App
run.py: Contains code for a Flask app deployment and the user interface used to predict results and display them.

