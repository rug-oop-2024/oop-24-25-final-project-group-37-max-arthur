from autoop.core.ml.metric import METRICS, get_metric
from autoop.functional.feature import detect_feature_types
from autoop.core.ml.model import REGRESSION_MODELS, CLASSIFICATION_MODELS, get_model

from autoop.core.ml.dataset import Dataset


import streamlit as st


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


def choose_model(model_type):
    if model_type == "Regression":
        model_options = REGRESSION_MODELS
    elif model_type == "Classification":
        model_options = CLASSIFICATION_MODELS
    selected_model = st.selectbox("Select a model", model_options)
    return get_model(selected_model)


def determine_task_type(target_feature):
    if target_feature.type == "categorical":
        return "Classification"
    elif target_feature.type == "numerical":
        return "Regression"


def choose_metrics():
    selected_metrics = st.multiselect("Select metrics", METRICS)
    metrics = [get_metric(m) for m in selected_metrics]
    return metrics, selected_metrics


def choose_target_column(dataset: 'Dataset'):
    target_column = st.selectbox("Select the target column", dataset.read().columns)
    st.write(f"Target column selected: {target_column}")
    return target_column


def choose_input_columns(dataset: 'Dataset', target_column: str):
    columns = [col for col in dataset.read().columns if col != target_column]
    input_columns = st.multiselect("Select input columns", columns)
    st.write(f"Input columns selected: {input_columns}")
    return input_columns


def generate_target_and_input_features(dataset, target_column: str, input_columns: list[str]):
    features = detect_feature_types(dataset)
    target_feature = next(feature for feature in features if feature.name == target_column)
    input_features = [feature for feature in features if feature.name in input_columns]
    return target_feature, input_features


def display_pipe_line_summary(dataset, target_column, input_columns, model_type, selected_model, metrics_names, dataset_split):
    with st.expander("Pipeline Summary"):
        st.write("### Dataset")
        st.write(f"**Name:** {dataset.name}")
        st.write(f"**Target Column:** {target_column}")
        st.write(f"**Input Columns:** {', '.join(input_columns)}")

        st.write("### Model")
        st.write(f"**Type:** {model_type}")
        st.write(f"**Selected Model:** {selected_model}")

        st.write("### Metrics")
        st.write(f"**Selected Metrics:** {', '.join(metrics_names)}")

        st.write("### Dataset Split")
        st.write(f"**Training Set:** {dataset_split * 100}%")
        st.write(f"**Testing Set:** {(1 - dataset_split) * 100}%")


def display_pipeline_results(results):
    with st.expander("## Results", expanded=True):
        st.write("### Metrics:")
        st.write(results['metrics'])
        st.write("### Predictions:")
        st.write(results['predictions'])
