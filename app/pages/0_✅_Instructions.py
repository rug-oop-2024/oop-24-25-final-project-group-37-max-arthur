import streamlit as st

st.set_page_config(
    page_title="Instructions",
    page_icon="👋",
)

st.markdown(open("app/content/instructions.md").read())
