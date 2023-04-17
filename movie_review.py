import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import streamlit as st
from sklearn.linear_model import LogisticRegression

# Load data from CSV file
df = pd.read_csv("https://drive.google.com/file/d/1-2xifmYspjb8f2TlE_K1jZ_RQnrqa27r/view?usp=sharing")

# Split data into train and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Extract review texts and labels from the data

test_texts = test_data["review"].tolist()
test_labels = test_data["sentiment"].tolist()


# Preprocess data
preprocessor = CountVectorizer(stop_words="english")
X_train = preprocessor.fit_transform(train_texts)
X_test = preprocessor.transform(test_texts)

# Train model
model = LogisticRegression()
model.fit(X_train, train_labels)

# Evaluate model
#y_pred = model.predict(X_test)
#print(classification_report(test_labels, y_pred))
#print("Accuracy: ", accuracy_score(test_labels, y_pred))


# Save the trained model and preprocessor
joblib.dump(model, "model_new1.pkl")
joblib.dump(preprocessor, "preprocessor_new1.pkl")



#creating a function to rerun the app 
def movie_review():
    
    st.header('WE REVIEW BETTER ;-D ')
    #using image to display it in the web app using pillow lib
    image_link = ("https://www.behindwoods.com/tamil-movie-reviews/reviews-2/images/review.jpg")
    st.image(image_link,width=500)
    #displaying info what this website is all about
    st.text("Are you someone who relies on movie reviews before watching a film?\n If so, then look no further!") 
    st.text ("If you find reading movie reviews tedious and prefer a quicker verdict on whether\na movie is worth watching, simply copy and paste the review here and we'll let you know.")
    st.markdown('We are happy to help you!')
    #taking input from the user
    client_input=st.text_input(' paste here ').lower()
    # model code here
    # Preprocess user input
    user_input_processed = preprocessor.transform([client_input])
    # Predict sentiment
    sentiment = model.predict(user_input_processed)[0]
    print(sentiment)
    if client_input:       
        if sentiment==("Negative"):     
            st.subheader(sentiment)
            st.warning('This is a warning', icon="⚠️")
        else:       
            st.subheader(sentiment+" You can go and watch it !")
            st.balloons() 

movie_review()

