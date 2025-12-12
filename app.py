import streamlit as st
import naive_bayes
import math
import re

st.set_page_config(page_title="Spam Detector", page_icon="📧")

@st.cache_resource
def get_model():
    #try to load model
    model_params = naive_bayes.load_model()
    
    if model_params:
        return model_params

    with st.spinner("Training model for the first time... (Downloading dataset)"):
        prior_probabilities, likelihoods, unique_words, all_words = naive_bayes.train_naive_bayes_model()
        naive_bayes.save_model(prior_probabilities, likelihoods, unique_words, all_words)
        return prior_probabilities, likelihoods, unique_words, all_words

def predict_message(text, prior_probabilities, likelihoods, unique_words, all_words):
    c_check_words = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    check_list = c_check_words.lower().split()

    spamicity_hamicity = {}

    for category in prior_probabilities:
        current_val = math.log(prior_probabilities[category])
        
        for word in check_list:
            if word in likelihoods[category]:
                current_val += math.log(likelihoods[category][word])
            else: 
                new_word_likelihood = 1 / (all_words + len(unique_words))
                current_val += math.log(new_word_likelihood)
        spamicity_hamicity[category] = current_val

    log_spam = spamicity_hamicity[1]
    log_ham = spamicity_hamicity[0]

    max_val = max(log_spam, log_ham)

    prob_spam = math.exp(log_spam - max_val)
    prob_ham = math.exp(log_ham - max_val)

    total = prob_spam + prob_ham
    spam_percentage = (prob_spam / total) * 100
    ham_percentage = (prob_ham / total) * 100

    if log_spam > log_ham:
        return "SPAM", "red", spam_percentage
    else:
        return "NOT SPAM", "green", spam_percentage

st.title("Email Spam Detector")
st.markdown("Paste an email below or upload a text file to check if it's **Spam** or **Ham**.")

#load the model
try:
    prior_probabilities, likelihoods, unique_words, all_words = get_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


tab1, tab2 = st.tabs(["Paste Text", "Upload File"])

input_text = None

with tab1:
    user_input = st.text_area("Email Content:", height=200, placeholder="Paste the email text here...")
    if st.button("Check Text"):
        input_text = user_input

with tab2:
    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    if uploaded_file is not None:
        #decode the uploaded file bytes to string
        input_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.info("File uploaded successfully!")

if input_text:
    prediction, color, confidence = predict_message(input_text, prior_probabilities, likelihoods, unique_words, all_words)
    
    st.divider()
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader(f":{color}[{prediction}]")
        st.metric("Confidence Score", f"{confidence:.1f}%")
        
    with col2:
        st.write("Confidence Bar:")
        if prediction == "SPAM":
            st.progress(int(confidence), text="Spam Probability")
        else:
            st.progress(int(confidence), text="Safety Probability")
elif input_text == "":
    st.warning("Please enter some text to analyze.")
