from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import speech_recognition as sr
import gradio as gr
import openai
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

# Load and process data
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
docs = load_data(path)
data = get_chunks(docs)


def store_data(data, embeddings):
  # vector store
  db = FAISS.from_documents(data, embeddings)
  return db

embeddings = embed(data, 'cpu', 'sentence-transformers/all-MiniLM-L6-v2')
db = store_data(data, embeddings)


llm = ChatOpenAI(model = "gpt-4o")

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
### Statefully manage chat history ###

messages_history = {}

def get_session_history(session_id: str):
    if session_id not in messages_history:
        messages_history[session_id] = ChatMessageHistory()
    return messages_history[session_id]

retriever=db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

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


# --- Response Generation ---
def generate_response(query):

    return conversational_rag_chain.invoke({"input": query}, config={"configurable": {"session_id": "1"}})["answer"]

# -- Gradio Interface --
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Your Input"),
    outputs=gr.Textbox(label="WiChat Response"),
    title="WiChat - Worldbank Ideas Chatbot"
)

def clear_chat_history():
    """Clears the chat history."""
    # Reset the chat history for session '1'
    if "1" in messages_history:
        messages_history["1"].clear()
    return [], gr.update(value="") # Return empty history and clear input box


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# WiChat - Worldbank Ideas Chatbot")
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Your Input")
    clear = gr.Button("Clear Chat")

    def respond(message, chat_history):
        # Use the generate_response function which uses the RAG chain and session history
        bot_message = generate_response(message)
        chat_history.append((message, bot_message))
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_chat_history, outputs=[chatbot, msg])
demo.launch(share=True)
