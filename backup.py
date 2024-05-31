import streamlit as st
from streamlit_chat import message
from langchain.llms import Ollama
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage

llm = Ollama(model="phi3")

class Deep_Gpt():
    def __init__(self, file_path):
        self.chat_history = []  
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        chunks = text_splitter.split_documents(documents=PyMuPDFLoader(file_path=file_path).load())
        
        embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device':'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        vectorstore = FAISS.from_documents(chunks, embedding_model)
        vectorstore.save_local("vectorstore")
        
        # Contextualizing prompt
        qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
{context}"""
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        # History-aware retriever
        self.history_aware_retriever = create_history_aware_retriever(llm, vectorstore.as_retriever(), qa_prompt)
        # Retrieval chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.history_aware_retriever, question_answer_chain)
        
    def query(self, question):
        response = self.rag_chain.invoke({"input": question, "chat_history": self.chat_history})
        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(HumanMessage(content=response["answer"]))
        return response["answer"]

oracle = Deep_Gpt(r"C:\Users\HAMMAD\Desktop\chatbot\pdfs\Generative_Deep_Learning.pdf")

st.set_page_config(page_title="Deep Learning Teacher", page_icon=":robot_face:", layout="wide", initial_sidebar_state="expanded")

st.markdown("<style>body {color: #ffffff; background-color: #121212;}</style>", unsafe_allow_html=True)

st.title("DeepTeacher")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

user_input = st.text_input("You:", "")

if user_input:
    response = oracle.query(user_input)
    st.session_state['messages'].append({"role": "user", "content": user_input})
    st.session_state['messages'].append({"role": "assistant", "content": response})

for msg in st.session_state['messages']:
    message(msg['content'], is_user=(msg['role'] == "user"))

st.markdown("---")
st.markdown("Made with ❤️ by Java")
