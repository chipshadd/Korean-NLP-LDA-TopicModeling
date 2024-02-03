# Open a command prompt
- Right click the kr_lda folder
- Click on 'Open in Terminal'

# When you want to analyze a set of articles in an excel file called 'export.xlsx'
# Copy the 'export.xlsx' file in the kr_lda folder and run the command below
# Note: Update the name of the excel file to what it is named.
python main.py analyze --file .\export.xlsx

# To see the list of topics and keywords in the LDA model:
python main.py info

# To train the model further with extra data, copy the training data into the kr_lda/training_data folder
# Everytime you run an analysis it will update the model so you do not need to train it with the data you are trying to analyze
python main.py train

