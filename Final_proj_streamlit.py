import streamlit as st
import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import easyocr
from PIL import Image
import pandas as pd
import numpy as np
import re
import mysql.connector
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image,ImageEnhance,ImageFilter,ImageOps,ImageDraw
import easyocr
from joblib import load
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import warnings
warnings.filterwarnings("ignore")
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from textblob import TextBlob
import spacy
from spacy import displacy
import nltk
import plotly.express as px
import plotly.graph_objects as go
nltk.download('punkt')
nltk.download('stopwords')

#python -m streamlit run Final_proj_streamlit.py

# SETTING PAGE CONFIGURATIONS
icon = Image.open("image1.jpg")
st.set_page_config(page_title="Final Project:| By sanjeev ",
                   page_icon=icon,
                   layout="wide",
                   initial_sidebar_state="expanded",
                   menu_items={'About': """# This web application is created to the model prediction, price prediction, Image processing and NLP *!"""})
st.markdown("<h1 style='text-align: center; color: Green;",
            unsafe_allow_html=True)

#st.snow
#python -m streamlit run Final_proj_streamlit.py


# CREATING OPTION MENU
with st.sidebar:   
    selected = option_menu(None, ["Home", "Customer_conversion","EDA", "Product_recommendation","NLP","Image"],
                       icons=["house", "cloud-upload", "pencil-square"],
                       default_index=0,
                       orientation="horizontal",
                       styles={"nav-link": {"font-size": "25px", "text-align": "centre", "margin": "-3px",
                                            "--hover-color": "#545454"},
                               "icon": {"font-size": "30px"},
                               "container": {"max-width": "5000px"},
                               "nav-link-selected": {"background-color": "#ff5757"}})

# HOME MENU
if selected == "Home":
    col1, col2 = st.columns(2)
    with col1:
        st.image(Image.open("image2.png"), width=500)
        st.markdown("## :green[**Technologies Used :**] Machine Learning,Python,easy OCR, Streamlit, , Pandas")
    with col2:
        st.write(
            '## This project is the comination of Machine Learning models, NLP, Complete EDA process and Image processing ')


# Customer conversion
data = pd.read_csv("classification_data.csv")

# Load the models inside the Streamlit app
if selected == "Customer_conversion":
    col1, col2 = st.columns(2)
    with col1:
    # Load the Decision Tree model
        with open('decision_tree_model.pkl', 'rb') as model_file:
             decision_tree_model = pickle.load(model_file)

        channelgrouplist = list(data['channelGrouping'].unique())
        channelgrouplist.sort()
        devices = list(data['device_deviceCategory'].unique())
        devices.sort()
        regions = list(data['geoNetwork_region'].unique())
        regions.sort()
        sources = list(data['latest_source'].unique())
        sources.sort()
        keyword = list(data['latest_keyword'].unique())
        keyword.sort()
        product_arr = list(data['products_array'].unique())
        product_arr.sort()

        device_deviceCategory = st.selectbox("Select Device Category", devices)
        geoNetwork_region = st.selectbox("Select GeoNetwork Region", regions)
        historic_session = st.number_input("Enter historic_session", min_value=0, value=0)
        historic_session_page = st.number_input("Enter historic_session_page", min_value=0, value=0)
        avg_session_time = st.number_input("Enter avg_session_time", min_value=0, value=0)
        avg_session_time_page = st.number_input("Enter avg_session_time_page", min_value=0, value=0)
        single_page_rate = st.number_input("Enter single_page_rate", min_value=0, value=0)
        sessionQualityDim = st.number_input("Enter sessionQualityDim", min_value=0, value=0)
        latest_visit_id = st.number_input("Enter latest_visit_id", min_value=0, value=0)
        latest_visit_number = st.number_input("Enter latest_visit_number", min_value=0, value=0)
        time_latest_visit = st.number_input("Enter time_latest_visit", min_value=0, value=0)
        avg_visit_time = st.number_input("Enter avg_visit_time", min_value=0, value=0)
        visits_per_day = st.number_input("Enter visits_per_day", min_value=0, value=0)
        latest_source = st.selectbox("Select Latest Source",sources)
        latest_medium = st.selectbox("Select Latest Medium", data['latest_medium'].unique())
        latest_keyword = st.selectbox("Enter Latest Keyword",keyword)
        latest_isTrueDirect = st.checkbox("Is True Direct", value=False)
        time_on_site = st.number_input("Enter time_on_site", min_value=0, value=0)
        products_array = st.selectbox("Enter product array",product_arr)
        transactionRevenue = st.number_input("Enter transactionRevenue", min_value=0, value=0)
        count_hit = st.number_input("Enter counthit")
        channelGrouping = st.selectbox("Enter channelGrouping",channelgrouplist)

        channels = int(channelgrouplist.index(channelGrouping))
        device = int(devices.index(device_deviceCategory))
        region = int(regions.index(geoNetwork_region))
        source = int(sources.index(latest_source))
        keywords = int(keyword.index(latest_keyword))
        product_arrr = int(product_arr.index(products_array))
    # Additional feature dictionary
    additional_feature = {
        "count_hit": count_hit,
        'channelGrouping':channels,
        'device_deviceCategory':device,
        'geoNetwork_region':region,
        'historic_session':historic_session,
        'historic_session_page':historic_session_page,
        'avg_session_time':avg_session_time,
        'avg_session_time_page':avg_session_time_page,
        'single_page_rate':single_page_rate,
        'sessionQualityDim':sessionQualityDim,
        'latest_visit_id':latest_visit_id,
        'latest_visit_number':latest_visit_number,
        'time_latest_visit':time_latest_visit,
        'avg_visit_time':avg_visit_time,
        'visits_per_day':visits_per_day,
        'latest_source':source,
        'latest_keyword':keywords,
        'latest_isTrueDirect':latest_isTrueDirect,
        'time_on_site':time_on_site,
        'transactionRevenue':transactionRevenue,
        'products_array':product_arrr
    }

    # Assuming all 21 features used during training are numerical
    all_features = [
            'count_hit', 'channelGrouping', 'device_deviceCategory',
        'geoNetwork_region', 'historic_session', 'historic_session_page',
        'avg_session_time', 'avg_session_time_page', 'single_page_rate',
        'sessionQualityDim', 'latest_visit_id', 'latest_visit_number',
        'time_latest_visit', 'avg_visit_time', 'visits_per_day',
        'latest_source', 'latest_keyword', 'latest_isTrueDirect',
        'time_on_site', 'transactionRevenue', 'products_array',  
        ]

    # Prediction button
    if st.button("Predict Conversion"):
        dff = pd.DataFrame([additional_feature])
        #dff = dff.apply(zscore)
        st.dataframe(dff)
        dt = decision_tree_model.predict(dff)

        st.write(dt)
        if  dt[0] == 0:
            st.write("Not converted")
        else:
            st.write("Converted")
    with col2:
        dat = {
    "Model": ["Logistic Regression", "k-Nearest Neighbors", "XGBoost", "Naive Bayes", "Decision Tree", "Random Forest"],
    "Accuracy": [0.90485, 0.98820, 0.99840, 0.80840, 0.96735, 0.99820],
    "Precision": [0.912178, 0.991220, 0.998641, 0.830782, 0.977112, 0.998143],
    "Recall": [0.880761, 0.985835, 0.998254, 0.733416, 0.967421, 0.998143],
    "F1 Score": [0.896194, 0.988520, 0.998447, 0.779068, 0.987266, 0.998143]
        }

    df = pd.DataFrame(dat)

# Display the DataFrame using Streamlit
    st.dataframe(df)


#Product_Recommendation

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

if selected == "Product_recommendation":

    # Create a sample dataset (replace this with your own dataset)
    data = {
        'Product': ['B001TH7GUU', 'B003ES5ZUU', 'B0019EHU8G', 'B006W8U2MU', 'B000QUUFRW','B000HPV3RW'],
        'Description': ['SanDisk', 'Metal Folding Portable Laptop Stand', 'Mediabridge HDMI Cable', 'Telephone Landline Extension Cord Cable', 'Mederma Stretch Marks Therapy','Lumino Cielo Stress Relief Therapy Exercise Squeeze Balls for Fingers']
    }

    df = pd.DataFrame(data)

    # Function to get recommendations
    def get_recommendations(user_input, df):
        df['UserInput'] = user_input
        df['Combined'] = df['Description'] + ' ' + df['UserInput']
        vectorizer = CountVectorizer().fit_transform(df['Combined'])
        similarity_matrix = cosine_similarity(vectorizer, vectorizer)
        indices = pd.Series(df.index, index=df['Product']).drop_duplicates()
        idx = indices[user_input]
        sim_scores = list(enumerate(similarity_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        top_recommendations = sim_scores[1:6]
        product_indices = [i[0] for i in top_recommendations]
        
        recommendations = df[['Product', 'Description']].iloc[product_indices].to_dict('records')
        return recommendations

    # Streamlit app
    def main():
        st.title("Product Recommendation")

        # User input
        user_input = st.text_input("Enter a product:","B0019EHU8G")

        # Get recommendations on button click
        if st.button("Get Recommendations"):
            recommendations = get_recommendations(user_input, df)
            st.success("Recommended Products:")
            for rec in recommendations:
                st.write(f"Product: {rec['Product']}, Description: {rec['Description']}")

    if __name__ == "__main__":
        main()


import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import easyocr
import numpy as np
# Image processing

if selected == "Image":
    
    def image():
        st.write("<h4 style='text-align:center; font-weight:bolder;'>Image Processing</h4>", unsafe_allow_html=True)
        upload_file = st.file_uploader('Choose a Image File', type=['png', 'jpg', 'webp'])

        if upload_file is not None:
            upload_image = np.asarray(Image.open(upload_file))
            u1 = Image.open(upload_file)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Read Original Image")
                st.image(upload_image,)
                width = st.number_input("**Enter Width**", value=(u1.size)[0])

            with col2:
                graysclae = u1.convert("L")
                st.subheader("Gray Scale Image")
                st.image(graysclae)
                height = st.number_input("**Enter Height**", value=(u1.size)[1])

        # Continue with the rest of your image processing logic...
            with col1:
                resize_image = u1.resize((int(width), int(height)))
                st.subheader("Resize Image")
                st.image(resize_image)
                radius = st.number_input("**Enter radius**", value=1)
                blur_org = u1.filter(ImageFilter.GaussianBlur(radius=int(radius)))
                st.subheader("Blurring with Original Image")
                st.image(blur_org)
                blur_gray = graysclae.filter(ImageFilter.GaussianBlur(radius=int(radius)))
                st.subheader("Blurring with Gray Scale Image")
                st.image(blur_gray)
                threshold = st.number_input("**Enter Threshold**", value=100)
                threshold_image = u1.point(lambda x: 0 if x < threshold else 255)
                st.subheader("Threshold Image")
                st.image(threshold_image)
                flip = st.selectbox("**Select Flip**", ["left-right", 'top-bottom'])
                st.subheader("Flipped Image")
                if flip == "left-right":
                    st.image(u1.transpose(Image.FLIP_LEFT_RIGHT))
                if flip == 'top-bottom':
                    st.image(u1.transpose(Image.FLIP_TOP_BOTTOM))
                brightness = st.number_input("**Enter Brightness**", value=1)
                st.subheader("Brightness Image")
                st.image((ImageEnhance.Brightness(u1)).enhance(int(brightness)))

            with col2:
                mirror_image = ImageOps.mirror(u1)
                st.subheader("Mirror Image")
                st.image(mirror_image)
                contrast = st.number_input("**Enter contrast**", value=1)
                contrast_org = ImageEnhance.Contrast(blur_org)
                st.subheader("Contrast with Original Image")
                st.image(contrast_org.enhance(int(contrast)))
                contrast_gray = ImageEnhance.Contrast(blur_gray)
                st.subheader("Contrast with Gray Scale Image")
                st.image(contrast_gray.enhance(int(contrast)))
                rotation = st.number_input("**Enter Rotation**", value=180)
                st.subheader("Rotation Image")
                st.image(u1.rotate(int(rotation)))
                sharpness = st.number_input("**Enter Sharness**", value=1)
                st.subheader("Sharpness Image")
                st.image((ImageEnhance.Sharpness(u1)).enhance(int(sharpness)))
                image_type = st.selectbox("**Select Image**", ["Original image", 'Gray Scale Image', "Blur Image",
                                                            "Threshold Image", "Sharpness Image", "Brightness Image"])

                if image_type == "Original image":
                    st.subheader("Edge Detection with Original Image")
                    st.image(u1.filter(ImageFilter.FIND_EDGES))
                if image_type == 'Gray Scale Image':
                    st.subheader("Edge Detection with Grayscale Image")
                    st.image(graysclae.filter(ImageFilter.FIND_EDGES))
                if image_type == "Blur Image":
                    st.subheader("Edge Detection with Blur Original Image")
                    st.image(blur_org.filter(ImageFilter.FIND_EDGES))

                if image_type == "Threshold Image":
                    st.subheader("Edge Detection with Threshold Image")
                    st.image(threshold_image.filter(ImageFilter.FIND_EDGES))
                if image_type == "Sharpness Image":
                    st.subheader("Edge Detection with Sharpness Image")
                    st.image(((ImageEnhance.Sharpness(u1)).enhance(int(sharpness))).filter(ImageFilter.FIND_EDGES))
                if image_type == "Brightness Image":
                    st.subheader("Edge Detection with Brightness Image")
                    st.image(((ImageEnhance.Brightness(u1)).enhance(int(brightness))).filter(ImageFilter.FIND_EDGES))

            reader = easyocr.Reader(['en'])
            bounds = reader.readtext(upload_image)
            if bounds:
                st.subheader("Extracted Text")
                file_name = upload_file.name
                if file_name == '1.png':

                    address, city = map(str, (bounds[6][1]).split(', '))
                    state, pincode = map(str, (bounds[8][1]).split())
                    image1_data = {
                        'Company': bounds[7][1] + ' ' + bounds[9][1],
                        'Card_holder_name': bounds[0][1],
                        'Desination': bounds[1][1],
                        'Mobile': bounds[2][1],
                        'Email': bounds[5][1],
                        'URL': bounds[4][1],
                        'Area': address[0:-1],
                        'City': city[0:-1],
                        'State': state,
                        'Pincode': pincode
                    }
                    st.json(image1_data)

            # Continue with other conditions...
    image()


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from wordcloud import WordCloud
from textblob import TextBlob
import spacy
from spacy import displacy

if selected == "NLP":
    
    def nlp_preprocess(text):
        # Tokenization
            words = word_tokenize(text)

            # Remove stopwords
            stop_words = set(stopwords.words('english'))
            filtered_words = [word.lower() for word in words if word.lower() not in stop_words]

            # Stemming
            porter_stemmer = PorterStemmer()
            stemmed_words = [porter_stemmer.stem(word) for word in filtered_words]

            return filtered_words, stemmed_words

    # Function for Keyword Extraction
    def extract_keywords(text):
        words = word_tokenize(text)
        
        # Assuming keywords are the most frequent non-stopwords
        stop_words = set(stopwords.words('english'))
        filtered_words = [word.lower() for word in words if word.lower() not in stop_words]

        # Count word frequencies
        word_freq = nltk.FreqDist(filtered_words)

        # Get the top 5 most frequent words as keywords
        keywords = [word for word, freq in word_freq.most_common(5)]

        return keywords

    # Function for Sentiment Analysis
    def perform_sentiment_analysis(text):
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        return sentiment_score

    # Function for Named Entity Recognition (NER)
    def perform_ner(text):
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(text)
        ner_output = displacy.render(doc, style='ent', jupyter=False)
        return ner_output

    def perform_sentiment_analysis(text):
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity

        # Categorize sentiment
        if sentiment_score > 0:
            sentiment_label = "Positive"
        elif sentiment_score < 0:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        return sentiment_label

    # Function for Word Cloud
    def generate_word_cloud(text):
        if text:
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            st.image(wordcloud.to_image())
        else:
            st.warning("Please enter some text to generate a word cloud.")

    # Streamlit App
    def main():
        st.title("NLP Streamlit App")

        # Tabs
        tabs = ["Stemming", "NER", "Word Cloud","Sentiment Analysis"]
        selected_tab = st.sidebar.radio("Select Tab", tabs)    

        # Stemming Tab
        if selected_tab == "Stemming":
            st.subheader("Stemming")
            input_for_stemming = st.text_area("Enter Text for Stemming:")
            if st.button("Process Stemming"):
                # Perform stemming on the provided text
                _, stemmed_words = nlp_preprocess(input_for_stemming)
                st.write("Stemmed Words:")
                st.write(stemmed_words)

        # NER Tab
        elif selected_tab == "NER":
            st.subheader("Named Entity Recognition (NER)")
            input_for_ner = st.text_area("Enter Text for NER:")
            if st.button("Process NER"):
                # Perform Named Entity Recognition on the provided text
                st.write("NER Output:")
                st.markdown(perform_ner(input_for_ner), unsafe_allow_html=True)

        # Word Cloud Tab
        elif selected_tab == "Word Cloud":
                st.subheader("Word Cloud")
                input_for_wordcloud = st.text_area("Enter Text for Word Cloud:")
                if st.button("Generate Word Cloud"):
                    # Generate word cloud for the provided text
                    generate_word_cloud(input_for_wordcloud)

        elif selected_tab == "Sentiment Analysis":
            st.subheader("Sentiment Analysis")
            input_for_sentiment = st.text_area("Enter Text for Sentiment Analysis:")
            if st.button("Analyze Sentiment"):
                # Analyze sentiment for the provided text
                sentiment_label = perform_sentiment_analysis(input_for_sentiment)
                st.write("Sentiment Label:")
                st.write(sentiment_label)

    if __name__ == "__main__":
        main()


if selected == "EDA":
    def load_data(file_path):
        print(f"Loading data from: {file_path}")
        data = pd.read_csv(file_path)
        print("Data loaded successfully.")
        return data

    
# Function to perform EDA
    def perform_eda(data):
        st.title("Exploratory Data Analysis (EDA)")

        # Display the first few rows of the dataset
        st.subheader("Preview of the Data")
        st.dataframe(data.head())

        # Summary statistics
        st.subheader("Summary Statistics")
        st.table(data.describe())

        # Data Information
        st.subheader("Data Information")
        st.table(data.info())

        # Univariate Analysis
        st.subheader("Univariate Analysis")

        # Histogram
        selected_column = st.selectbox("Select a column for histogram:", data.columns)
        fig_hist = px.histogram(data, x=selected_column, title=f'Histogram of {selected_column}')
        st.plotly_chart(fig_hist)

        # Boxplot
        fig_box = px.box(data, x=selected_column, title=f'Boxplot of {selected_column}')
        st.plotly_chart(fig_box)

        # Bivariate Analysis
        st.subheader("Bivariate Analysis")

        # Scatter plot
        x_axis = st.selectbox("Select X-axis for scatter plot:", data.columns)
        y_axis = st.selectbox("Select Y-axis for scatter plot:", data.columns)
        fig_scatter = px.scatter(data, x=x_axis, y=y_axis, title=f'Scatter Plot: {x_axis} vs {y_axis}')
        st.plotly_chart(fig_scatter)

        # Correlation matrix
        st.subheader("Correlation Matrix")
        fig_corr = px.imshow(data.corr(), color_continuous_scale='viridis', title='Correlation Matrix')
        st.plotly_chart(fig_corr)

        # Multivariate Analysis
        st.subheader("Multivariate Analysis")

        # Pair plot
        fig_pair = px.scatter_matrix(data, title='Pair Plot')
        st.plotly_chart(fig_pair)

# Main function
    def main():
        st.write(page_title="EDA with Streamlit", layout="wide")

        st.sidebar.title("EDA & Report")

        # Upload dataset
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

        if uploaded_file is not None:
            data = load_data(uploaded_file)
            perform_eda(data)

    if __name__ == "__main__":
            main()
