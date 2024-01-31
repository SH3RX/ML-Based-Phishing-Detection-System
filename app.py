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
    st.write('For this educational project, '
             'I created my own data set and defined features, some from the literature and some based on manual analysis. '
             'I used requests library to collect data, BeautifulSoup module to parse and extract features. ')

    st.subheader('Data set')
    st.write('I used _"phishtank.org"_ & _"tranco-list.eu"_ as data sources.')
    st.write('Totally 24228 websites ==> **_12001_ legitimate** websites | **_12227_ phishing** websites')
    st.write('Data set was created in January 2024.')

    # ----- THE PIE CHART ----- #
    labels = 'Phishing', 'Legitimate'
    phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
    legitimate_rate = 100 - phishing_rate
    sizes = [phishing_rate, legitimate_rate]
    explode = (0.1, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
    ax.axis('equal')
    st.pyplot(fig)
    # ----- !!!!! ----- #

    st.write('Features + URL + Label ==> Dataframe')
    st.markdown('Label is 1 for phishing, 0 for legitimate')
    number = st.slider("Select row number to display", 0, 100)
    st.dataframe(ml.legitimate_df.head(number))

    @st.cache
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')

    csv = convert_df(ml.df)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv',
    )

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
    st.write('_	https://gobpeposts.top/uQFYwz/_')
    st.write('_https://att-mail-info-104416.weeblysite.com/_')
    st.write('_https://i3d.net.ar/BT/BancoInternacional.html_')
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
