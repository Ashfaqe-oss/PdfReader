import os

from langchain.llms import OpenAI

import streamlit as st

from langchain.document_loaders import PyPDFLoader

from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)

from config import OPENAI_API_KEY

os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

llm = OpenAI(temperature=0.1, verbose=True)

# uploaded_file = st.file_uploader("Choose a file")

# if uploaded_file is not None:

loader = PyPDFLoader('annualreport.pdf')

pages = loader.load_and_split()

# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, collection_name='annualreport')


vectorstore_info = VectorStoreInfo(
    name="annual_report",
    description="a banking annual report as a pdf",
    vectorstore=store
)

# Convert the document store into a langchain toolkit
toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LangChain
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

st.title('GPT Reader - for now on preprovided doc')
# Create a text input box for the user
prompt = st.text_input('Input your prompt here')

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    response = agent_executor.run(prompt)
    # ...and write it out to the screen
    st.write(response)

    # With a streamlit expander  
    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content) 