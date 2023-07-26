import streamlit as st
import pickle
import os
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader    
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# sidebar contents
with st.sidebar:
    st.title('LLM Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    * [Streamlit](https://streamlit.io/)
    * [LangChain](https://python.langchain.com/)
    * [OpenAI](https://platform.openai.com/docs/models) LLM model
 
    ''')
    add_vertical_space(5)
    st.write('Made by Ayushi Malaviya [Tech Lover](https://github.com/Ayushimalaviya)')

def main():
    st.header("Chat with PDF")

    load_dotenv()

    # upload a pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        st.write(pdf.name)
        pdf_reader = PdfReader(pdf)
        # st.write(pdf_reader)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        # st.write(text)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        # st.write(chunks)
        
        #embeddings
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", 'rb') as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings Loaded from the disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write('Embeddings Computations completed')
        # Accept user question/query
        query = st.text_input("Ask question about the pdf file.")
        # st.write(query)
        
        if query:
            docs = VectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(model_name='gpt-3.5-turbo')
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)
        


if __name__ == '__main__':
    main()