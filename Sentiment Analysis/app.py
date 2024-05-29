from keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st

model = load_model('model.h5')

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


def predict_sentiment(review):
  sequence = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequence, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment

def main():
   st.title('Sentiment Analysis')
   input = st.text_input('Enter the Review : ')
   output = predict_sentiment(input)
   if input != "":
    if output == "positive":
        st.success('The sentiment of the review is: {}'.format(output))
    else:
        st.error('The sentiment of the review is: {}'.format(output))

if __name__ == "__main__":
    main()