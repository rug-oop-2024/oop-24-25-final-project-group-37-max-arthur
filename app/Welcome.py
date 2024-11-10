import streamlit as st
import sys
import os

# Add the project root directory to the system path
current_dir = os.path.dirname(__file__)

sys.path.append(os.path.abspath(os.path.join(current_dir, '..')))


st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
st.sidebar.success("Select a page above.")
st.markdown(open("README.md").read())
