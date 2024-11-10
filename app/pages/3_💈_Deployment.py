import streamlit as st
import app.core.deployment_handler as dph

from app.core.system import AutoMLSystem
from app.core.modelling_handler import display_pipeline_summary


def render_deployment():
    st.set_page_config(page_title="Deployment", page_icon="💈")
    st.write("# Deployment")

    automl = AutoMLSystem.get_instance()

    pipelines = automl.registry.list(type="pipeline")

    if not pipelines:
        st.write("No pipelines found.")
        return

    with st.container(border=True):
        st.write("### Select a Pipeline")
        pipeline, name, selected_pipeline_artifact = dph.select_pipeline(
            pipelines)
        model = pipeline._model

        dph.delete_pipeline_button(automl,
                                   name + " Pipeline",
                                   selected_pipeline_artifact.id)

        display_pipeline_summary(pipeline, name + " Summary")

    dph.prediction_accordion(automl, pipeline, model)


if __name__ == "__main__":
    render_deployment()
