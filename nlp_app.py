import spacy
import pandas as pd
from gensim.summarization import summarize
import streamlit as st 
import spacy_streamlit
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nlp = spacy.load("en_core_web_sm")

st.set_page_config("NLP Basics")
with st.sidebar:
    st.markdown(""" **Developed by** [M.Arslan Akram](https://www.linkedin.com/in/arslanakram1/)
    """)
    st.markdown(""" **Source Code ** [Github](https://github.com/MuhammadArslanAkram/basic_nlp)
    """)
    st.header("Navigation")
    nav_list=["Tokenization","Name Entity Recognition","Sentence Segmentation","Sentiment Analysis","Summary"]
    choice=st.radio("Go to",nav_list)
    st.header("About App")
    st.info(
        """This App uses State of the Art Spacy Library along with Python.It uses Streamlit
           for implemention of beatiful and easy web app.
        """)

st.markdown(""" ## Natural Language Processing using **SPACY** """)
raw_text=st.text_area("Text here")
doc=nlp(raw_text)

if raw_text is not None:

    if choice == "Tokenization":
        if st.button("Tokenize"):
            spacy_streamlit.visualize_tokens(doc=doc,attrs=["text","pos_","dep_","lemma_","shape_"])
    
    if choice == "Name Entity Recognition":
        if st.button("Analyze"):
            spacy_streamlit.visualize_ner(doc=doc,labels=nlp.get_pipe("ner").labels)

    if choice == "Sentence Segmentation":
        l=[]
        if st.button("Segmentize"):
            st.write(f""" There are **{len(list(doc.sents))} Sentences** in this text dataset.""")
            for sent in doc.sents:
                 l.append(sent)
            d={"Sentences":l}
            df=pd.DataFrame(data=d)
            st.write(df)

    if choice == "Sentiment Analysis":
        sid=SentimentIntensityAnalyzer()
        b=sid.polarity_scores(raw_text)
        
        if (b["compound"]>=0.5 and b["compound"]<1):
            st.write(f""" Polarity score for this text data is **{b["compound"]}** 
                          showing **Strong Postive** Sentiment""")

        elif (b["compound"]>=0.1 and b["compound"]<0.5):
            st.write(f""" Polarity score for this text data is **{b["compound"]}** 
                          showing **Postive** Sentiment""")

        elif b["compound"]<0:
            st.write(f""" Polarity score for this text data is **{b["compound"]}** 
                          showing **Negative** Sentiment""")
        
        else:
            st.write(f""" Polarity score for this text data is **{b["compound"]}** 
                          showing **Neutral** Sentiment""")

    if choice == "Summary":
        sum_words_count=st.slider(label="Words in Summary",min_value=50,max_value=500,step=25,value=100)
        if st.button("Summarize"):
            req_text=summarize(text=raw_text,word_count=sum_words_count,)
            st.write(req_text)

else:
    pass
    
