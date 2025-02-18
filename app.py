
import openai
import os
import pyttsx3
import streamlit as st
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

api_key = st.secrets["OPENAI_API_KEY"]

def load_data(path):
    loader1 = DirectoryLoader(path, glob='*.txt', show_progress=True)
    docs = loader1.load()
    return docs

def get_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = text_splitter.split_documents(docs)
    return chunks

# embed data sources
def embed(data, device, model):
  model_kwargs = {'device': device}
  encode_kwargs = {'normalize_embeddings': False}

  embeddings = HuggingFaceEmbeddings(
    model_name = model,
    model_kwargs = model_kwargs,
    encode_kwargs = encode_kwargs
  )
  return embeddings


path = 'scraped_data'


def store_data(data, embeddings):
  # vector store
  db = FAISS.from_documents(data, embeddings)
  return db


# cache the model
@st.cache_resource
def load_model():
    return ChatOpenAI(model="gpt-4o")

llm = load_model()

# Define the chat prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are called WiChat, which is short for Worldbank Ideas Chatbot, the chatbot for the Worldbank Ideas Project. You are friendly and follow instructions to answer questions extremely well. Please be truthful and give direct answers. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the response short and concise in at most five sentences. If the user chats in a different language, translate accurately and respond in the same language. You will provide specific details and accurate answers to user queries on the Worldbank Ideas Project."),
        MessagesPlaceholder("chat_history"),
        ("human", "Use only the retrieved {context} to answer the user question {input}.")
    ]
)

# --- Create RAG chain ---

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)



def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.chat_history:
        st.session_state.chat_history[session_id] = ChatMessageHistory()
    return st.session_state.chat_history[session_id]


# --- Response Generation ---
def generate_response(query):
    retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
    return conversational_rag_chain.invoke({"input": query}, config={"configurable": {"session_id": "1"}})["answer"]

logo = "Wichat2.png"

# Startup Screen
def startup_screen():
    st.title("WiChat")
    st.subheader("Version 1.0")
    st.image("Wichat2.png")
    st.spinner("Loading...")  # Spinner while loading
    time.sleep(5)
    st.session_state.current_screen = "main"  # Switch to main screen after loading
    st.rerun()
    

# Function to display the menu
def show_menu():
    """Displays the menu options using a selectbox."""
    menu_options = ["Clear Chat", "Help"]
    selected_option = st.selectbox("Menu Options", menu_options)
    if selected_option:
        menu_callback(selected_option)

# Function to display the account options
def show_account():
    """Displays the account options (Login/SignUp) using a selectbox."""
    account_options = ["Login", "SignUp"]
    selected_option = st.selectbox("Account Options", account_options)
    if selected_option:
        account_callback(selected_option)

# Function to handle menu callback actions
def menu_callback(option):
    """Handles actions based on the menu option selected."""
    if option == "Clear Chat":
        clear_chat()
    elif option == "Help":
        st.info("Help: Contact support for assistance.")



# Function to record audio
def record_audio():
    """Records audio, converts it to text, and updates the input field."""
    audio_value = st.audio_input("record a voice message to transcribe")

    if audio_value:
      transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file = audio_value
      )

    user_input = transcript.text
    return user_input

# Function to speak text
def speak_text(response):
    """Converts text to speech using pyttsx3."""
    engine = pyttsx3.init()
    engine.say(response)
    engine.runAndWait()

# Function to send a message to chatbot
def send_message(user_input):
    """Sends a message to the chatbot and updates the chat history."""
    if prompt := st.chat_input(user_input):
    # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Generate response from the chatbot
        response = st.write(generate_response(user_input))
        with st.chat_message("assistant"):
            st.markdown(response)
        # Clear the input field
        st.session_state.user_input = ""

        
# Function to clear the chat history
def clear_chat():
    """Clears the chat history."""
    st.session_state.chat_history = {}

# ------------------------- Streamlit UI -------------------------
def main():

    # Initialize session state

    if 'data' not in st.session_state:
        
         st.session_state.data =  get_chunks(load_data(path))
    
    if 'db' not in st.session_state:
         embeddings = embed(st.session_state.data, 'cpu', 'sentence-transformers/all-MiniLM-L6-v2')
         st.session_state.db = store_data(st.session_state.data, embeddings)
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = {}

    if 'current_screen' not in st.session_state:
      startup_screen()
    elif st.session_state.current_screen == "startup":
      startup_screen()
    elif st.session_state.current_screen == "login":
      login_screen()


    # Render UI based on the current screen
    if st.session_state.current_screen == "main":
        st.title("WiChat")
        st.logo(logo, size="medium")


        # Welcome Card
        st.markdown("<div style='padding: 10px; border-radius: 10px;'><h4>Welcome!!! What would you like to know? </h4></div>", unsafe_allow_html=True)

      
        if st.button("Menu"):
                show_menu()
      

        # Input section
        input_col1, input_col2, input_col3 = st.columns([8, 1, 1], vertical_alignment="bottom")

        with input_col1:
            user_input = st.text_input("", placeholder="Type your message...", key="input_text")

        with input_col2:
            if st.button("🎤", key="mic_button"):
                
                user_input = record_audio()
                response = send_message(user_input)
                speak_text(response)
        with input_col3:
            if st.button("Send"):
                send_message(user_input)


    elif st.session_state.current_screen == "login":
        # Render login screen
        login_screen()
    elif st.session_state.current_screen == "signup":
        # Render signup screen
        signup_screen()
    st.markdown(
    """
    <style>
    .stTextInput {
        margin-top: -10px;
    }
    .stButton {
        margin-top: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
    )
if __name__ == "__main__":
    main()
