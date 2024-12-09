import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

import streamlit as st
data=pd.read_csv(r'C:\Users\HP\Downloads\spam.csv')
data['Category']=data['Category'].replace(['ham','spam'],['Not spam','Spam'])
mess=data['Message']
cat=data['Category']
#training model
(mess_train,mess_test,cat_train,cat_test)=train_test_split(mess,cat,test_size=0.2)
cv=CountVectorizer(stop_words='english')
features=cv.fit_transform(mess_train)
#creating model
model=MultinomialNB()
model.fit(features,cat_train)
#testing model
features_test=cv.transform(mess_test)
#print(model.score(features_test,cat_test))
#predict the data
def predict(message):

    input_message=cv.transform([message]).toarray()

    result=model.predict(input_message)
    return result

Output=predict('Dear,will call tomorow.plsaccomodate')
print(Output)


