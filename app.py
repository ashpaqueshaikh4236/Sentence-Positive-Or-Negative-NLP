import nltk
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
ps = PorterStemmer()

def stemming(content):
    if not content.strip():
        return ''
    
    stemmed_content = re.sub('[^a-zA-Z]',' ',content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Sentiment Analysis")

input_sms = st.text_area("Enter the message",' ')

if st.button('Predict'):
    if not input_sms.strip() :
        st.warning('Please Write Something!')
    else:
        # 1. Preprocess
        transformed_sms = stemming(input_sms)

        # 2. vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3.predict
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            st.success("Postive")
        else:
            st.warning("Negative")
