import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import re
import csv
import nltk
import string
from nlp_id.lemmatizer import Lemmatizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nlp_id.stopword import StopWord

st.header('Sentiment Analysis')
with st.expander('Analyze CSV'):
    neg = st.file_uploader('Negatif Lexicon')
    pos = st.file_uploader('Positif lexicon')
    upl = st.file_uploader('Upload file')

    def case_folding(text):
        text = text.lower()
        # Menghapus tanda baca
        text = re.sub(r'[^\w\s]', ' ', text)
        #Menghapus angka
        text = re.sub(" \d+", '', text)
        # Menghapus white space
        text = text.strip('')
        # menghapus tab, garis baru dan backslice
        text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ")
        # Menghapus non ASCII (emoticon, chinese word, .etc)
        text = text.encode('ascii', 'replace').decode('ascii')
        text = text.translate(str.maketrans(" "," ",string.punctuation))
        text = re.sub('\s+',' ',text)
        return text

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    lemmatizer = Lemmatizer()

    stopword = StopWord()

    slang_dictionary = pd.read_csv('https://raw.githubusercontent.com/nikovs/data-science-portfolio/master/topic%20modelling/colloquial-indonesian-lexicon.csv')
    slang_dict = pd.Series(slang_dictionary['formal'].values,index=slang_dictionary['slang']).to_dict()

    def Slangwords(text):
        for word in text.split():
            if word in slang_dict.keys():
                text = text.replace(word, slang_dict[word])
        return text

    def case_folding_final(text):
        #Menghapus angka
        text = re.sub(" \d+", '', text)
        # Menghapus white space
        text = text.strip()
        return text

    def tokenization(teks):
        text_list = []
        for txt in teks.split(" "):
            text_list.append(txt)
        return text_list

    def preprocess_text(df):
        df['review_processed'] = ''
        for i, row in df.iterrows():
            text = row['Review']
            clean_text = case_folding(text)
            clean_text = stemmer.stem(clean_text)
            clean_text = lemmatizer.lemmatize(clean_text)
            clean_text = stopword.remove_stopword(clean_text)
            clean_text = Slangwords(clean_text)
            clean_text = case_folding_final(clean_text)
            clean_text = tokenization(clean_text)
            df['review_processed'][i] = clean_text
        return df

    def sentiment_analysis_lexicon_indonesia(text):
        score = 0
        for word in text:
            if (word in list_positive):
                score += 1
        for word in text:
            if (word in list_negative):
                score -= 1
        polarity=''
        if (score > 0):
            polarity = 'positive'
        elif (score < 0):
            polarity = 'negative'
        else:
            polarity = 'neutral'
        return score, polarity
#
    if neg:
        df_negative = pd.read_csv(neg, header=None)
        list_negative = list(df_negative.loc[::,0])

#
    if pos:
        df_positive = pd.read_csv(pos, header=None)
        list_positive = list(df_positive.loc[::,0])

#
    if upl:
        #import uploaded data
        df = pd.read_excel(upl)
        #preprocess data
        df_clean= preprocess_text(df)
        #sentiment analysis
        hasil = df_clean['review_processed'].apply(sentiment_analysis_lexicon_indonesia)
        hasil = list(zip(*hasil))
        df_clean['polarity_score'] = hasil[0]
        df_clean['polarity'] = hasil[1]
        #distribution chart
        color = ['#CDFAD5', '#F6FDC3', '#FF8080']
        name = df_clean['polarity'].unique()
        label=df_clean.polarity.value_counts()
        fig1, ax1 = plt.subplots()
        text_prop = {'family':'monospace', 'fontsize':'small', 'fontweight':'light'}
        ax1.pie(label, labels=name, colors=color, autopct='%1.1f%%',
                shadow=False, startangle=90, textprops=text_prop)
        ax1.axis('equal')
        st.pyplot(fig1)
        st.write(label)
        st.write(df)


        @st.cache_data
        def convert_df(df):
            return df.to_csv()

        csv = convert_df(df_clean)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='sentiment.csv',
            mime='text/csv',
        )
