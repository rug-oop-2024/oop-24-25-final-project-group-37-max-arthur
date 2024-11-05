import streamlit as st
import pickle

from app.core.system import AutoMLSystem

st.set_page_config(page_title="Deployment", page_icon="💈")
st.write("# Deployment")

# Create a page where you can see existing saved pipelines.
# Allow the user to select existing pipelines and based on the selection show a pipeline summary.
# Once the user loads a pipeline, prompt them to provide a CSV on which they can perform predictions.

automl = AutoMLSystem.get_instance()

pipelines = automl.registry.list(type="pipeline")
st.write(f"Available Pipelines:{pickle.loads(pipelines[0].data)}")
