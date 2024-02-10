import streamlit as st
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.retrievers.web_research import WebResearchRetriever
import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
from langchain.chat_models import ChatOpenAI
from langchain.utilities import GoogleSearchAPIWrapper
from typing import Dict
import logging
from functools import wraps
import time

# Set Streamlit page config
st.set_page_config(page_title="HelpIT Explorer", page_icon="🌐")

# Function to apply retry mechanism with exponential backoff
def retry(exception_to_check, tries=4, delay=3, backoff=2):
    """Retry decorator with exponential backoff."""
    def deco_retry(f):
        @wraps(f)
        def f_retry(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return f(*args, **kwargs)
                except exception_to_check as e:
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return f(*args, **kwargs)
        return f_retry
    return deco_retry

# Settings function to configure LangChain components
def settings():
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore_public = FAISS(embeddings_model.embed_query, index, InMemoryDocstore({}), {})
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0, streaming=True)
    search = GoogleSearchAPIWrapper()
    web_retriever = WebResearchRetriever.from_llm(
        vectorstore=vectorstore_public,
        llm=llm,
        search=search,
        num_search_results=5
    )
    return web_retriever, llm

# Callback handler classes for LangChain integration
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.info(self.text)

class PrintRetrievalHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container.expander("Context Retrieval")

    def on_retriever_start(self, query: str, **kwargs):
        self.container.write(f"**Question:** {query}")

    def on_retriever_end(self, documents, **kwargs):
        for idx, doc in enumerate(documents):
            source = doc.metadata["source"]
            self.container.write(f"**Results from {source}**")
            self.container.text(doc.page_content)

# Function to validate and refine answers before presentation
def validate_and_refine_answer(answer: str, criteria: Dict[str, any]) -> str:
    if len(answer) < criteria.get('min_length', 100):
        return "Je suis désolé, je n'ai pas pu trouver une réponse suffisamment détaillée. Pouvez-vous préciser votre question ?"
    return answer

# Instruction for handling user questions, including answer validation
def handle_user_question(question: str, llm, web_retriever, criteria: Dict[str, any]) -> None:
    if not question.strip():
        st.warning('Veuillez entrer une question valide.')
        return
    
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm, retriever=web_retriever)
    retrieval_streamer_cb = PrintRetrievalHandler(st.container())
    stream_handler = StreamHandler(st.empty(), initial_text="`Réponse:`\n\n")
    result = qa_chain({"question": question}, callbacks=[retrieval_streamer_cb, stream_handler])
    
    refined_answer = validate_and_refine_answer(result['answer'], criteria)
    st.info('`Réponse:`\n\n' + refined_answer)
    st.info('`Sources:`\n\n' + result['sources'])

# Main application flow
if 'retriever' not in st.session_state:
    st.session_state['retriever'], st.session_state['llm'] = settings()

web_retriever, llm = st.session_state['retriever'], st.session_state['llm']
question = st.text_input("`Posez-moi une question:`")

answer_criteria = {
    "min_length": 100,  # Minimum length of the answer
    # Add more criteria as needed
}

if question:
    handle_user_question(question, llm, web_retriever, answer_criteria)
