import streamlit as st

import app.core.modelling_handler as mh
from app.core.dataset_handler import choose_dataset
from app.core.system import AutoMLSystem
from autoop.core.ml.pipeline import Pipeline


def render_modelling():
    """
    Renders the Modelling page for the Streamlit application.
    This function sets up the page configuration, displays the modelling
    section header, and provides an interface for designing a machine learning
    pipeline. Users can select a dataset, choose target and input columns,
    determine the model type, select a model, configure the dataset split, and
    choose evaluation metrics. The function also handles the execution and
    display of the pipeline results.

    Args:
        None
    Raises:
        ValueError: If any required elements are missing for pipeline creation.
    Returns:
        None
    """

    st.set_page_config(page_title="Modelling", page_icon="📈")

    st.write("# ⚙ Modelling")

    mh.write_helper_text("In this section, you can design a machine learning"
                         " pipeline to train a model on a dataset.")

    automl = AutoMLSystem.get_instance()

    datasets = automl.registry.list(type="dataset")

    with st.container(border=True):
        st.write("### Pipeline Configuration")

        selected_dataset = choose_dataset(datasets)

        selected_target_column = mh.choose_target_column(selected_dataset)

        selected_input_columns = mh.choose_input_columns(
            selected_dataset, selected_target_column)

        target_feature, input_features = mh.generate_target_and_input_features(
            selected_dataset, selected_target_column, selected_input_columns)

        model_type = mh.determine_task_type(
            target_feature, selected_dataset.read()[target_feature.name])
        st.write(f"Model type determined: {model_type}")
        if model_type == "Classification":
            target_feature.type = "categorical"

        selected_model = mh.choose_model(model_type)

        dataset_split = st.slider("Select the dataset split",
                                  0.1, 0.9, 0.8, 0.1)

        selected_metrics = mh.choose_metrics(model_type)

        required_elements = [
            selected_dataset, selected_metrics, target_feature,
            input_features, selected_model, dataset_split
        ]

        if all(required_elements):

            if 'pipeline' not in st.session_state:
                pipeline = Pipeline(
                    dataset=selected_dataset,
                    metrics=selected_metrics,
                    target_feature=target_feature,
                    input_features=input_features,
                    model=selected_model,
                    split=dataset_split)
                st.session_state.pipeline = pipeline

            mh.execute_pipeline_button(st.session_state.pipeline)

            if 'results' in st.session_state:
                mh.display_pipeline_results(st.session_state.results,
                                            st.session_state.pipeline)

            mh.display_pipeline_summary(st.session_state.pipeline)

            if st.session_state.pipeline.model.parameters:
                mh.save_pipeline_button(automl, st.session_state.pipeline)
        else:
            st.session_state.clear()


if __name__ == "__main__":
    render_modelling()
