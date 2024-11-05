import streamlit as st
import pandas as pd
import pickle

from autoop.core.ml.artifact import Artifact
from app.core.system import AutoMLSystem
from app.core.modelling_handler import choose_metrics, determine_task_type, choose_target_column, choose_model, generate_target_and_input_features, choose_input_columns, display_pipe_line_summary, write_helper_text, display_pipeline_results
from app.core.dataset_handler import choose_dataset, ask_for_input
from autoop.core.ml.pipeline import Pipeline


def render_modelling():
    st.set_page_config(page_title="Modelling", page_icon="📈")

    st.write("# ⚙ Modelling")

    write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

    automl = AutoMLSystem.get_instance()

    datasets = automl.registry.list(type="dataset")

    selected_dataset = choose_dataset(datasets)

    selected_target_column = choose_target_column(selected_dataset)

    selected_input_columns = choose_input_columns(selected_dataset, selected_target_column)

    target_feature, input_features = generate_target_and_input_features(selected_dataset, selected_target_column, selected_input_columns)

    model_type = determine_task_type(target_feature)
    st.write(f"Model type determined: {model_type}")

    selected_model = choose_model(model_type)

    dataset_split = st.slider("Select the dataset split", 0.1, 0.9, 0.8, 0.1)

    # Prompt the user to select a set of compatible metrics. Should we determine based on which model type which metrics are most useful?
    selected_metrics, metrics_names = choose_metrics()

    display_pipe_line_summary(selected_dataset, selected_target_column, selected_input_columns, model_type, selected_model, metrics_names, dataset_split)

####################### Pipeline Save ############################
    def save_pipeline(pipeline):
        pip_art = pipeline.to_artifact(name="test")
        model_artifact = next(artifact for artifact in pipeline.artifacts if artifact.name.startswith("pipeline_model"))
        automl.registry.register(model_artifact)
        automl.registry.register(pip_art)
######################## Pipeline Save ############################

    if selected_dataset and selected_metrics and target_feature and input_features and selected_model and dataset_split:
        pipeline = Pipeline(dataset=selected_dataset, metrics=selected_metrics, target_feature=target_feature, input_features=input_features, model=selected_model, split=dataset_split)
        if st.button("Execute Pipeline"):
            results = pipeline.execute()
            display_pipeline_results(results)
        for artifact in pipeline.artifacts:
            bytes = artifact.read()
            st.write(f"Artifacts: {artifact.name} {pickle.loads(bytes)}")
        if st.button("Save Pipeline"):
            save_pipeline(pipeline)  # it works but implement check to check weather it already exists






# artifact .data is binary. How can we save these artifacts in data base. How can we reference them back for instantiation of pipeline artifact

    # Prompt the user to give a name and version for the pipeline and convert it into an artifact which can be saved.
    # with st.expander("Save Pipeline"):
    #     pipeline_name = ask_for_input("Pipeline Name")
    #     pipeline_version = ask_for_input("Pipeline Version")
    #     if st.button("Save Pipeline"):

    #         artifacts = pipeline._artifacts
    #         # artifact = Artifact(name=pipeline_name, data=selected_dataset.data, asset_path=selected_dataset.asset_path ,version=pipeline_version, type="pipeline", metadata={"model": str(selected_model), "input_features": [str(f) for f in input_features], "target_feature": str(target_feature), "dataset_split": dataset_split, "metrics": selected_metrics})
    #         for artifact in artifacts:
    #             automl._registry.register(artifact)
    #         st.success("Pipeline saved successfully!")

if __name__ == "__main__":
    render_modelling()
