import streamlit as st
import machine_learning as ml
import features_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt

# Streamlit app content
st.title('Phishing Website Detection using Machine Learning')
st.write('This ML-based app is developed to detect phishing websites only using content data. Not URL!')

with st.expander("PROJECT DETAILS"):
    st.subheader('Approach')
    st.write('I used Supervised Learning to classify phishing and legitimate websites. '
             'The detection content-based approach is to focus on HTML of the websites. '
             )

    st.subheader('Data set')
    st.write('I used _"phishtank.org"_ & _"tranco-list.eu"_ as data sources.')
    st.write('Totally 24228 websites ==> **_12001_ legitimate** websites | **_12227_ phishing** websites')
    st.write('Data set was created in January 2024.')

    # Adding title and legend
    ax.set_title('Phishing vs Legitimate Websites')
    ax.legend(labels, loc='upper right')

    st.pyplot(fig)

    st.write('Features + URL + Label ==> Dataframe')
    st.markdown('Label is 1 for phishing, 0 for legitimate')

    st.subheader('Results')
    st.write('I used 5 different ML classifiers of scikit-learn and tested them implementing k-fold cross validation.'
             'Firstly obtained their confusion matrices, then calculated their accuracy, precision and recall scores.'
             'Comparison table is below:')
    
    st.table(ml.df_results)
    st.write('RF --> Random Forest')
    st.write('SVM --> Support Vector Machine')
    st.write('DT --> Decision Tree')
    st.write('AB --> AdaBoost')
    st.write('NB --> Gaussian Naive Bayes')

with st.expander('EXAMPLE PHISHING URLs:'):
    st.write('_https://keepo.io/upate_')
    st.write('_https://yoshiiremo.net/op1/tus_')
    st.write('_https://hotm.art/c5Z8ty_')
    st.caption('INFORMATION: PHISHING WEB PAGES HAVE SHORT LIFECYCLE!')

# Model selection
choice = st.selectbox("Please select your machine learning model",
                 [
                     'Random Forest', 'Support Vector Machine', 'Decision Tree', 'AdaBoost',
                     'Gaussian Naive Bayes',
                 ]
                )

model = ml.nb_model

if choice == 'Random Forest':
    model = ml.rf_model
    st.write('RF model is selected!')
elif choice == 'Support Vector Machine':
    model = ml.svm_model
    st.write('SVM model is selected!')
elif choice == 'Decision Tree':
    model = ml.dt_model
    st.write('DT model is selected!')
elif choice == 'AdaBoost':
    model = ml.ab_model
    st.write('AB model is selected!')
elif choice == 'Gaussian Naive Bayes':
    model = ml.nb_model
    st.write('NB model is selected!')

# URL input and check button with color modification
url = st.text_input('Enter the URL', key='url_input', help='e.g., https://example.com')

# Apply custom styling to the input box
st.markdown("""
    <style>
        div[data-baseweb="input"] {
            background-color: #e6f7ff;
            border-color: #0056b3;
            border-radius: 5px;
        }
        div[data-baseweb="input"] input {
            color: #0056b3;
        }
    </style>
""", unsafe_allow_html=True)

if st.button('Check!', key='check_button'):
    try:
        response = re.get(url, verify=False, timeout=4)
        if response.status_code != 200:
            st.error(f"HTTP connection was not successful for the URL: {url}")
        else:
            soup = BeautifulSoup(response.content, "html.parser")
            vector = [fe.create_vector(soup)]  # it should be a 2D array, so I added []
            result = model.predict(vector)
            if result[0] == 0:
                st.success("This web page seems legitimate!")
                st.balloons()
            else:
                st.warning("Attention! This web page is a potential PHISHING!")
                st.snow()

    except re.exceptions.RequestException as e:
        st.error(f"Error: {e}")
