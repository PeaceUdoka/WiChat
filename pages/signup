# signup Screen

import streamlit as st

def signup_screen():

    st.title("Sign Up")

    # Input fields for signup
    username = st.text_input("Username")
    password = st.text_input("Password")
    confirm_password = st.text_input("Confirm Password")
    email = st.text_input("Email")

    signup_button = st.button("Sign Up")

    if signup_button:
        if password == confirm_password:

            st.success(f"Account created successfully for {username} with email {email}!")
            # navigate back to the chat screen or login screen
            st.session_state.current_screen = "login"
            st.rerun()
        else:
            st.error("Passwords do not match. Please try again.")

    back_button = st.button("Back to Login")
    if back_button:
        st.session_state.current_screen = "login"
        st.rerun()
signup_screen()
