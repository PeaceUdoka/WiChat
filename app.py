

from kivymd.app import MDApp
from kivymd.uix.screenmanager import MDScreenManager
from kivymd.uix.label import MDLabel
from kivymd.uix.spinner import MDSpinner
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.screen import Screen
from kivy.clock import Clock
from kivy.uix.widget import Widget
from kivymd.uix.button import MDIconButton, MDRaisedButton, MDFlatButton
from kivymd.uix.card import MDCard
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.textfield import MDTextField
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.list import MDList
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
import speech_recognition as sr
from kivy.uix.image import Image
from kivy.clock import mainthread
from kivymd.uix.dialog import MDDialog
from kivy.utils import get_color_from_hex
from langchain.chat_models import ChatOpenAI
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

import pyttsx3
import os
#os.environ['OPENAI_API_KEY'] = key

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

# Startup Screen
class StartupScreen(Screen):
    def on_enter(self):
        layout = BoxLayout(orientation='vertical', spacing=10, pos_hint={'center_x': 0.5, 'center_y': 0.5}, size_hint=(None, None), size=("300dp", "400dp"))
        # Image at the center
        image = Image(source="Wichat2.png", size_hint=(None, None), size=("500dp", "300dp"), pos_hint={'center_x': 0.5})
        self.spinner = MDSpinner(size_hint=(None, None), size=(50, 50), pos_hint={'center_x': 0.5, 'center_y': 1}, active=True)
        title = MDLabel(text="WiChat", font_style="H4", halign="center", bold=True)
        version = MDLabel(text="Version 1.0", font_style="Caption", halign="center")

        layout.add_widget(image)
        layout.add_widget(self.spinner)
        layout.add_widget(title)
        layout.add_widget(version)
        self.add_widget(layout)

        Clock.schedule_once(self.switch_to_main, 40)  # Switch to main screen after 2.5 seconds

    def switch_to_main(self, dt):
        self.manager.current = "main"






# Login Screen
class LoginScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Outer BoxLayout to center the form
        outer_layout = BoxLayout(orientation="vertical", spacing=10, size_hint=(None, None),
                                 size=("300dp", "250dp"))
        outer_layout.pos_hint = {"center_x": 0.5, "center_y": 0.5}  # Centering

        self.username_input = MDTextField(hint_text="Username", size_hint_x=1)
        self.password_input = MDTextField(hint_text="Password", password=True, size_hint_x=1)

        login_button = MDRaisedButton(text="Login", size_hint=(1, None), height=40)
        login_button.bind(on_release=self.login)

        back_button = MDFlatButton(text="Back", size_hint=(1, None), height=40)
        back_button.bind(on_release=self.back_to_chat)

        outer_layout.add_widget(self.username_input)
        outer_layout.add_widget(self.password_input)
        outer_layout.add_widget(login_button)
        outer_layout.add_widget(back_button)

        self.add_widget(outer_layout)

    def login(self, instance):
        print("Logging in with username and password:", self.username_input.text, self.password_input.text)

    def back_to_chat(self, instance):
        self.manager.current = "main"


# Signup Screen
class SignupScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Outer BoxLayout to center the form
        outer_layout = BoxLayout(orientation="vertical", spacing=10, size_hint=(None, None), size=("300dp", "350dp"))
        outer_layout.pos_hint = {"center_x": 0.5, "center_y": 0.5}  # Centering

        # Title
        title = MDLabel(text="Sign Up", halign="center", font_style="H4", size_hint_y=None, height=50)
        outer_layout.add_widget(title)

        # Username
        self.username_input = MDTextField(hint_text="Username", size_hint_x=1)
        outer_layout.add_widget(self.username_input)

        # Email
        self.email_input = MDTextField(hint_text="Email", size_hint_x=1)
        outer_layout.add_widget(self.email_input)

        # Password
        self.password_input = MDTextField(hint_text="Password", password=True, size_hint_x=1)
        outer_layout.add_widget(self.password_input)

        # Sign Up Button
        signup_button = MDRaisedButton(text="Sign Up", size_hint=(1, None), height=40)
        signup_button.bind(on_release=self.signup)
        outer_layout.add_widget(signup_button)

        # Back Button
        back_button = MDFlatButton(text="Back", size_hint=(1, None), height=40)
        back_button.bind(on_release=self.back_to_chat)
        outer_layout.add_widget(back_button)

        self.add_widget(outer_layout)

    def signup(self, instance):
        print("Signing up with username, email, and password:", self.username_input.text, self.email_input.text, self.password_input.text)

    def back_to_chat(self, instance):
        self.manager.current = "main"







# Main Chatbot Screen
class ChatScreen(Screen):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mic_button = MDIconButton(text="🎤")  # Example of defining the mic_button
        layout = BoxLayout(orientation='vertical')
        self.card_color = (1, 1, 1, 1)
        self.engine = pyttsx3.init()  # Initialize text-to-speech engine


        # Toolbar
        toolbar = BoxLayout(size_hint_y=None, height="65dp", padding="10dp", spacing="10dp")
        self.menu_button = MDIconButton(icon="menu")
        self.menu_button.bind(on_release=self.show_menu)

        title = MDLabel(text="WiChat", halign="left", bold=True, font_size="20sp")
        self.account_button = MDIconButton(icon="account")
        self.account_button.bind(on_release=self.show_account)

        toolbar.add_widget(self.menu_button)
        toolbar.add_widget(title)
        toolbar.add_widget(self.account_button)
        layout.add_widget(toolbar)

        # Welcome Card
        md_card = MDCard(size_hint=(None, None), size=("700dp", "70dp"),
                         pos_hint={"center_x": 0.5, "center_y": 0.8}, elevation=5)
        md_label = MDLabel(text="Welcome!!! Ask me anything about the WorldBank IDEAS project!",
                           halign="center", theme_text_color="Secondary", bold=True)
        md_card.add_widget(md_label)
        layout.add_widget(md_card)

        # Chat Display
        self.chat_scroll = MDScrollView()
        self.chat_list = MDList()
        self.chat_scroll.add_widget(self.chat_list)
        layout.add_widget(self.chat_scroll)

        # Input and Send Button
        input_container = BoxLayout(size_hint=(None, None), size=("350dp", "50dp"),
                                    pos_hint={"center_x": 0.5, "y": 0.02}, spacing="10dp",)
        self.input_text = MDTextField(hint_text="Ask WiChat anything...", size_hint_x=0.7, multiline=False)

        # microphone
        mic_button = MDIconButton(icon="microphone", size_hint_x=0.1, on_release=self.record_audio)
        input_container.add_widget(mic_button)


        send_button = MDRaisedButton(text="Send", size_hint_x=0.3, on_release=self.send_message)
        input_container.add_widget(self.input_text)
        input_container.add_widget(send_button)
        layout.add_widget(input_container)

        # Dropdown Menus
        self.menu_items = [
            {"viewclass": "OneLineListItem", "text": "Toggle Theme",
             "on_release": lambda: self.menu_callback("Toggle Theme")},
            {"viewclass": "OneLineListItem", "text": "Clear Chat",
             "on_release": lambda: self.menu_callback("Clear Chat")},
            {"viewclass": "OneLineListItem", "text": "Help", "on_release": lambda: self.menu_callback("Help")},
            {"viewclass": "OneLineListItem", "text": "Choose Card Color", "on_release": self.open_color_dialog},
        ]
        self.menu = MDDropdownMenu(caller=self.menu_button, items=self.menu_items, width_mult=4)

        self.account_items = [
            {"viewclass": "OneLineListItem", "text": "Login", "on_release": lambda: self.account_callback("Login")},
            {"viewclass": "OneLineListItem", "text": "SignUp", "on_release": lambda: self.account_callback("SignUp")},
        ]
        self.account = MDDropdownMenu(caller=self.account_button, items=self.account_items, width_mult=4)

        self.add_widget(layout)

    def open_color_dialog(self):
        """Opens a color selection dialog where users can pick a card color."""

        color_map = {
            "Red": "#FF0000",
            "Green": "#00FF00",
            "Blue": "#0000FF",
            "Yellow": "#FFFF00",
            "Purple": "#800080",
            "Orange": "#FFA500",
            "Black": "#000000",
            "White": "#FFFFFF",
            "Gray": "#808080",
        }

        # Container for color buttons
        color_grid = MDGridLayout(cols=3, padding=10, spacing=10, adaptive_height=True)

        for color_name, hex_value in color_map.items():
            btn = MDRaisedButton(
                text=color_name,
                md_bg_color=get_color_from_hex(hex_value),
                on_release=lambda btn, c=color_name: self.set_card_color(c)
            )
            color_grid.add_widget(btn)

        # BoxLayout to ensure content is inside the dialog
        content_layout = MDBoxLayout(orientation="vertical", adaptive_height=True)
        content_layout.add_widget(color_grid)

        self.color_dialog = MDDialog(
            title="Pick a Card Color",
            type="custom",
            content_cls=content_layout,  # Ensure content is fully inside the dialog
            buttons=[
                MDRaisedButton(text="Close", on_release=lambda x: self.color_dialog.dismiss())
            ],
        )
        self.color_dialog.open()

    def set_card_color(self, color_name):
        """Set the card color based on the selected color name."""
        color_map = {
            "Red": "#FF0000",
            "Green": "#00FF00",
            "Blue": "#0000FF",
            "Yellow": "#FFFF00",
            "Purple": "#800080",
            "Orange": "#FFA500",
            "Black": "#000000",
            "White": "#FFFFFF",
            "Gray": "#808080",
        }

        self.card_color = color_map.get(color_name, "#FFFFFF")  # Default to white
        self.apply_card_color()
        self.color_dialog.dismiss()

    def apply_card_color(self):
        """Applies the selected card color to all chat messages, including the welcome card."""
        color = get_color_from_hex(self.card_color)  # Convert HEX to RGBA
        for child in self.chat_list.children:
            if isinstance(child, MDCard):
                child.md_bg_color = color

        # Also apply color to the welcome message card
        if hasattr(self, "welcome_card"):
            self.welcome_card.md_bg_color = color

        # Apply color to all cards in the app
        for screen in self.manager.screens:
            for widget in screen.walk():  # Recursively get all widgets
                if isinstance(widget, MDCard):
                    widget.md_bg_color = color


    def show_menu(self, obj):
        self.menu.open()

    def show_account(self, obj):
        self.account.open()

    def menu_callback(self, option):
        if option == "Toggle Theme":
            self.toggle_theme()
        elif option == "Clear Chat":
            self.clear_chat()
        elif option == "Help":
            print("Help Button Pressed")
        self.menu.dismiss()

    def account_callback(self, option):
        if option == "Login":
            self.manager.current = "login"
        elif option == "SignUp":
            self.manager.current = "signup"
        self.account.dismiss()

    def record_audio(self, obj):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            self.mic_button.icon = "microphone-off"
            self.speak_text("Listening...")  # Feedback using TTS

            recognizer.adjust_for_ambient_noise(source, duration=1)  # Reduce background noise
            try:
                audio = recognizer.listen(source, timeout=5)
                print("Audio captured!")  # Debugging output
                text = recognizer.recognize_google(audio)
                print(f"Recognized Text: {text}")  # Debugging output
                self.input_text.text = text  # Update text field
            except sr.RequestError:
                text = recognizer.recognize_sphinx(audio)
                self.input_text.text = text
            except sr.UnknownValueError:
                print("Could not understand audio.")  # Debugging output
                self.update_text_input("Could not understand audio.")
            except sr.RequestError as e:
                print(f"Speech Recognition service error: {e}")  # Debugging output
                self.update_text_input("Error with speech recognition service.")

            self.mic_button.icon = "microphone"  # Reset icon

    @mainthread
    def update_text_input(self, text):
        self.input_text.text = text

    def speak_text(self, text):
        """ Convert text to speech """
        self.engine.say(text)
        self.engine.runAndWait()




    def send_message(self, obj):
        message = self.input_text.text.strip()
        if message:
            #user_card = MDCard(size_hint= (None, None), size = ("300dp", 50), pos_hint={"right": 1}, padding=10, elevation=5, md_bg_color=self.card_color)
            # User message on the right
            msg_label = MDLabel(text=message, halign="right", size_hint_x=0.8, padding=70, theme_text_color="Primary")
            #user_card.add_widget(msg_label)

            user_box = BoxLayout(orientation="horizontal", size_hint_y=None, height=180)
            user_box.add_widget(Widget())
            #user_box.add_widget(user_card)
            self.chat_list.add_widget(user_box)


            self.chat_list.add_widget(msg_label)


            #self.chat_list.add_widget(user_card)
            self.input_text.text = "" # Clear input after sending


            response = generate_response(message)
            #response_card = MDCard(size_hint= (None, None), size = ("300dp", 50), pos_hint={"left": 1}, padding=10, elevation=5, md_bg_color=self.card_color)
            # WiChat response on the left
            response_label = MDLabel(text=response, halign="left", size_hint_x=0.8, padding=50, theme_text_color="Secondary")
            #response_card.add_widget(response_label)

            response_box = BoxLayout(orientation="horizontal", size_hint_y=None, height=180)
            #response_box.add_widget(response_card)
            response_box.add_widget(Widget())

            self.chat_list.add_widget(response_box)
            #self.chat_list.add_widget(response_card)

            self.chat_list.add_widget(response_label)

            # Scroll to the bottom after sending a message
            Clock.schedule_once(lambda dt: setattr(self.chat_scroll, 'scroll_y', 0))


    def clear_chat(self):
        self.chat_list.clear_widgets()

    def clear_memory(self):
        global memory
        memory.clear()
        print("Chat memory cleared.")

    def toggle_theme(self):
        MDApp.get_running_app().theme_cls.theme_style = "Dark" if MDApp.get_running_app().theme_cls.theme_style == "Light" else "Light"


# WiChat App
class WiChat(MDApp):
    def build(self):
        sm = MDScreenManager()
        sm.add_widget(StartupScreen(name="startup"))
        sm.add_widget(ChatScreen(name="main"))
        sm.add_widget(LoginScreen(name="login"))
        sm.add_widget(SignupScreen(name="signup"))
        return sm


if __name__ == "__main__":
    WiChat().run()
