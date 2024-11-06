import streamlit as st
import pickle

from app.core.system import AutoMLSystem
from app.core.modelling_handler import display_pipeline_summary
from app.core.dataset_handler import upload_csv_button


def render_deployment():
    st.set_page_config(page_title="Deployment", page_icon="💈")
    st.write("# Deployment")

    automl = AutoMLSystem.get_instance()

    pipelines = automl.registry.list(type="pipeline")

    if not pipelines:
        st.write("No pipelines found.")
        return

    name = st.selectbox(
        "Select a pipeline",
        [pipeline_artifact.name for pipeline_artifact in pipelines]
                        )

    selected_pipeline_artifact = next(
        (p for p in pipelines if p.name == name), None
        )

    pipeline = pickle.loads(selected_pipeline_artifact.read())

    st.write(f"### Selected Pipeline: {name}")

    display_pipeline_summary(pipeline, name)

    if st.button(f"Delete {name}"):
        automl.registry.delete(selected_pipeline_artifact.id)
        st.success(f"Pipeline {name} was deleted successfully.")

    csv = upload_csv_button()
# Once the user loads a pipeline, prompt them to provide a CSV on which they can perform predictions.


if __name__ == "__main__":
    render_deployment()
