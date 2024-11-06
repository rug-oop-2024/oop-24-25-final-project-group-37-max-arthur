from autoop.core.ml.metric import METRICS, get_metric, Metric
from autoop.functional.feature import detect_feature_types, Feature
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS
from autoop.core.ml.model import get_model, Model
from autoop.core.ml.pipeline import Pipeline
from app.core.system import AutoMLSystem


from autoop.core.ml.dataset import Dataset
from app.core.dataset_handler import ask_for_input

import streamlit as st


def write_helper_text(text: str) -> None:
    """
    Writes the given text to the Streamlit app with a specific style.
    Args:
        text (str): The text to be written in the Streamlit app.
    Returns:
        None
    """
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def choose_model(model_type: str) -> 'Model':
    """
    Choose a machine learning model based on the specified model type.
    Args:
        model_type (str): The type of model to choose. Can be either
        "Regression" or "Classification".
    Returns:
        Model: An instance of the selected machine learning model.
    """
    if model_type == "Regression":
        model_options = REGRESSION_MODELS
    elif model_type == "Classification":
        model_options = CLASSIFICATION_MODELS
    selected_model = st.selectbox("Select a model", model_options)
    return get_model(selected_model)()


def determine_task_type(target_feature: 'Feature') -> str:
    """
    Determines the type of machine learning task based on the target feature.
    Args:
        target_feature (Feature): The target feature for which the task type
        is to be determined.
                                  It should have an attribute 'type' which can
                                  be either 'categorical' or 'numerical'.
    Returns:
        str: The type of task, either 'Classification' if the target feature
        is categorical, or 'Regression' if the target feature is numerical.
    """
    if target_feature.type == "categorical":
        return "Classification"
    elif target_feature.type == "numerical":
        return "Regression"


def choose_metrics() -> list['Metric']:
    """
    Prompts the user to select metrics from a predefined list and returns the
    selected metrics.
    Returns:
        list['Metric']: A list of selected metrics.
    """
    selected_metrics = st.multiselect("Select metrics", METRICS)
    metrics = [get_metric(m) for m in selected_metrics]
    return metrics


def choose_target_column(dataset: 'Dataset') -> str:
    """
    Prompts the user to select a target column from the given dataset using a
    select box.
    Args:
        dataset (Dataset): The dataset object from which to select the target
        column.
    Returns:
        str: The name of the selected target column.
    """
    target_column = st.selectbox(
        "Select the target column", dataset.read().columns
        )
    st.write(f"Target column selected: {target_column}")
    return target_column


def choose_input_columns(dataset: 'Dataset', target_column: str) -> list[str]:
    """
    Allows the user to select input columns from a dataset, excluding the
    target column.
    Args:
        dataset (Dataset): The dataset from which to select columns.
        target_column (str): The name of the target column to exclude from
        selection.
    Returns:
        list[str]: A list of selected input column names.
    """
    columns = [col for col in dataset.read().columns if col != target_column]
    input_columns = st.multiselect("Select input columns", columns)
    st.write(f"Input columns selected: {input_columns}")
    return input_columns


def generate_target_and_input_features(
    dataset: 'Dataset', target_column: str, input_columns: list[str]
     ) -> tuple['Feature', list['Feature']]:
    """
    Generates the target feature and input features from the given dataset.
    Args:
        dataset (Dataset): The dataset from which to extract features.
        target_column (str): The name of the target column.
        input_columns (list[str]): A list of column names to be used as input
        features.
    Returns:
        tuple[Feature, list[Feature]]: A tuple containing the target feature
        and a list of input features.
    """
    features = detect_feature_types(dataset)
    target_feature = next(
        feature for feature in features if feature.name == target_column
        )
    input_features = [
        feature for feature in features if feature.name in input_columns
        ]
    return target_feature, input_features


def display_pipeline_summary(pipeline: 'Pipeline', name: str = "Summary"
                             ) -> None:
    """
    Display a summary of the given pipeline using Streamlit.
    Args:
        pipeline (Pipeline): The pipeline object containing dataset, model,
        and metrics information.
        name (str, optional): The name to be displayed in the expander header.
        Defaults to "Summary".
    Returns:
        None
    """
    with st.expander(f"Pipeline {name}"):
        st.write("### Dataset")
        st.write(f"**Name:** {pipeline._dataset.name}")
        st.write(f"**Target Column:** {pipeline._target_feature}")
        st.write("**Input Columns:**")
        for feature in pipeline._input_features:
            st.write(f"{str(feature)}")

        st.write("### Model")
        st.write(f"**Type:** {pipeline.model._type}")
        st.write(f"**Selected Model:** {pipeline.model.__class__.__name__}")

        st.write("### Metrics")
        st.write(
            f"**Selected Metrics:**"
            f"{', '.join(str(m) for m in pipeline._metrics)}")

        st.write("### Dataset Split")
        st.write(f"**Training Set:** {pipeline._split * 100}%")
        st.write(f"**Testing Set:** {(1 - pipeline._split) * 100}%")


def display_pipeline_results(results: dict) -> None:
    """
    Display the pipeline results using Streamlit.
    Args:
        results (dict): A dictionary containing the results of the pipeline.
                        It should have the following keys:
                        - 'metrics': The performance metrics of the pipeline.
                        - 'predictions': The predictions made by the pipeline.
    Returns:
        None
    """
    with st.expander("## Results", expanded=True):
        st.write("### Metrics:")
        st.write(results['metrics'])
        st.write("### Predictions:")
        st.write(results['predictions'])


def execute_pipeline_button(pipeline: 'Pipeline') -> None:
    """
    Handles the execution of a pipeline when the "Execute Pipeline" button
    is pressed.
    Args:
        pipeline (Pipeline): An instance of the Pipeline class that contains
        the logic to be executed.
    Returns:
        None
    """

    if st.button("Execute Pipeline"):
        results = pipeline.execute()
        display_pipeline_results(results)


def save_pipeline_button(automl: 'AutoMLSystem', pipeline: 'Pipeline') -> None:
    """
    Displays a button to save a machine learning pipeline using Streamlit.
    This function renders a Streamlit interface to save a given machine
    learning pipeline. It prompts the user to input a name for the pipeline
    and provides a button to save the pipeline using pickle. Upon successful
    saving, a success message is displayed.
    Args:
        automl (AutoMLSystem): The AutoML system instance that manages the
        pipeline registry.
        pipeline (Pipeline): The machine learning pipeline to be saved.
    Returns:
        None
    """
    st.write("### Save Pipeline")
    name = ask_for_input("Pipeline Name")
    if st.button("Save Pipeline with pickle"):
        pipeline_artifact = pipeline.to_artifact(name=name)
        automl.registry.register(pipeline_artifact)
        st.success("Pipeline saved successfully!")
