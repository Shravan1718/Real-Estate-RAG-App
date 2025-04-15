import streamlit as st
import streamlit.components.v1 as components
from rag import process_urls, generate_answer

st.title("Real Estate Research Tool")

url1 = st.sidebar.text_input("URL_1")
url2 = st.sidebar.text_input("URL_2")
url3 = st.sidebar.text_input("URL_3")

placeholder = st.empty()

process_url_button = st.sidebar.button("Process URLs")

#Process the URLs
if process_url_button:
    urls = [url for url in (url1, url2,url3) if url != '']
    if len(urls)==0:
        placeholder.text("You must provide at least 1 valid url")
    else:
        for status in process_urls(urls):
            placeholder.text(status)

#Generate an answer for query 
query = placeholder.text_input("Question")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer:")
        st.write(answer)

        if sources:
            st.subheader("Sources:")
            for source in sources.split("\n"):
                st.write(source)
    except RuntimeError as e:
        placeholder.text("You must process URLs first")

#Storing sample URLs to test the App
with st.sidebar.expander("Sample URLs", expanded=False):
    sample_urls = [
        "https://timesofindia.indiatimes.com/business/international-business/rocket-companies-to-acquire-mr-cooper-in-9-4-billion-all-stock-deal/articleshow/119811298.cms",
        "https://timesofindia.indiatimes.com/india/delhi-police-arrest-man-for-fraudulent-sale-of-mortgaged-property-worth-crores/articleshow/119902471.cms"
    ]

    for url in sample_urls:
        st.code(url, language='text')
        components.html(f"""
            <button onclick="navigator.clipboard.writeText('{url}')">📋 Copy</button>
        """, height=30)