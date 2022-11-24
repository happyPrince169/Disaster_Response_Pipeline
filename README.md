# Disaster Response Pipeline Project

# Project overview

This is my project in Udacity Data Science course. In this project, I will create a ML model to classify messages that are received from a disaster, and a webapp where we can enter a message and get back the classified results.


# File structure of the project

- app

| - template

| |- master.html  # main page of web app

| |- go.html  # classification result page of web app

|- run.py  # Flask file that runs app


- data

|- disaster_categories.csv  # data to process

|- disaster_messages.csv  # data to process

|- process_data.py

|- InsertDatabaseName.db   # database to save clean data to

- models

|- train_classifier.py


- README.md

# Description

App folder contains templates folder and "run.py" for the web app.

Data folder contains the datasets and the scripts "process_data.py" for data cleaning and transfering.

Models folder including "train_classifier.py" and "classifier.pkl" for the ML model. Because of the file size limit of GitHub, I cannot push the pkl file to the repo.

README file

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run your web app: `python run.py`

4. Go to http://0.0.0.0:3001/ to open the homepage
