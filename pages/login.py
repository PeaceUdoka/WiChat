import streamlit as st

def login_screen():
    st.title("Login")

    # Add unique keys to each input field
    username = st.text_input("Username")
    password = st.text_input("Password")

    # Login button
    login_button = st.button("Login")

    if login_button:

        if username and password:
            st.success("Logged in successfully!")
            st.session_state.current_screen = "main"  # Switch to main screen on successful login
            st.rerun()
        else:
            st.error("Invalid credentials. Please try again.")

    back_button = st.button("Back")

    if back_button:
        st.session_state.current_screen = "startup"  # Go back to startup screen
        st.rerun()
login_screen()
