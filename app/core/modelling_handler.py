import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from torch import Tensor

from app.core.dataset_handler import ask_for_input
from app.core.deployment_handler import get_feature_names
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.metric import METRICS, Metric, get_metric
from autoop.core.ml.model import (CLASSIFICATION_MODELS, REGRESSION_MODELS,
                                  Model, get_model)
from autoop.core.ml.pipeline import Pipeline
from autoop.functional.feature import Feature, detect_feature_types


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


def determine_task_type(
        target_feature: 'Feature',
        target_column: pd.Series
) -> str:
    """
    Determines the type of machine learning task based on the target feature.

    Args:
        target_feature (Feature): The target feature for which the task type
        is to be determined.
        target_column (pd.Series): The column data of the target feature.

    Returns:
        str: The type of task, either 'Classification' if the target feature
        is categorical, or 'Regression' if the target feature is numerical.
    """
    target_column = target_column.to_numpy()
    if target_feature.type == "categorical":
        return "Classification"
    elif target_feature.type == "numerical":
        # give user option to interpret numerical integer target as categorical
        if np.issubdtype(target_column.dtype, np.integer):
            interpret_as_categorical = st.checkbox(
                "Interpret numerical target feature as categorical"
            )
            if interpret_as_categorical:
                return "Classification"
        return "Regression"


def choose_metrics(model_type: str) -> list['Metric']:
    """
    Prompts the user to select metrics from a predefined list and returns the
    selected metrics.

    Returns:
        list['Metric']: A list of selected metrics.
    """
    if model_type == "Regression":
        available_metrics = METRICS[:3]
    elif model_type == "Classification":
        available_metrics = METRICS[3:]
    selected_metrics = st.multiselect("Select metrics", available_metrics)
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
        "Select the target column", dataset.read().columns)
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
    select_all = st.checkbox("Select All Columns")

    if select_all:
        input_columns = columns  # Select all columns if checkbox is checked
    else:
        input_columns = st.multiselect("Select input columns", columns)

    with st.container(height=200, border=False):
        st.write(f"Input columns selected: {input_columns}")
    return input_columns


def generate_target_and_input_features(
        dataset: 'Dataset', target_column: str,
        input_columns: list[str]) -> tuple['Feature', list['Feature']]:
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
        feature for feature in features if feature.name == target_column)
    input_features = [f for f in features if f.name in input_columns]
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
        st.write(f"**Name:** `{pipeline._dataset.name}`")
        st.write(f"**Target Column:** `{pipeline._target_feature}`")
        st.write("**Input Columns:**")
        with st.container(height=200, border=True):
            for feature in pipeline._input_features:
                st.write(f"`{str(feature)}`")

        st.write("### Model")
        st.write(f"**Type:** `{pipeline.model._type}`")
        st.write(f"**Selected Model:** `{pipeline.model.__class__.__name__}`")
        st.write("**Parameters:**")
        if pipeline.model.parameters:
            with st.container(height=200, border=True):
                st.write(pipeline.model.parameters)
        else:
            st.write("Model has no parameters yet.")

        st.write("### Metrics")
        st.write(
            f"**Selected Metrics:** "
            f"`{', '.join(str(m) for m in pipeline._metrics)}`")

        st.write("### Dataset Split")
        col1, col2 = st.columns(2)
        p_s = pipeline._split
        with col1:
            st.write(f"**Train Set:** `{round((p_s * 100), 2)}%`")
        with col2:
            st.write(f"**Test Set:** `{round(((1 - p_s) * 100), 2)}%`")


def display_pipeline_results(results: dict, pipeline: 'Pipeline') -> None:
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
    with st.expander("## Results"):
        st.write("### Metrics:")
        st.write(results['metrics'])

        metrics = results.get('metrics', {})
        st.write("### Predictions:")
        predictions = results.get('predictions', None)
        if predictions is not None:
            if isinstance(predictions, Tensor):
                predictions = predictions.numpy()
            feature_names = get_feature_names(pipeline)

            data = pipeline._compact_vectors(pipeline._test_X)

            df = pd.DataFrame(data, columns=feature_names)
            df['Predictions'] = predictions

            st.write(df)
        else:
            st.write("No predictions available.")
        if all(isinstance(v, dict) for v in metrics.values()):
            metric_names = list(metrics.keys())
            train_values = [v.get('train', 0) for v in metrics.values()]
            test_values = [v.get('test', 0) for v in metrics.values()]

            bar_width = 0.35
            indices = range(len(metric_names))

            fig, ax = plt.subplots(figsize=(10, 6))

            bars_train = ax.bar([i - bar_width/2 for i in indices],
                                train_values, width=bar_width, label='Train',
                                color='skyblue')

            bars_test = ax.bar([i + bar_width/2 for i in indices], test_values,
                               width=bar_width, label='Test', color='salmon')

            ax.set_ylabel('Scores')
            ax.set_title('Model Performance Metrics')
            ax.set_xticks(indices)
            ax.set_xticklabels(metric_names, rotation=45, ha='right')
            ax.legend()
            plt.tight_layout()

            # Add annotations
            for bar in bars_train:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

            for bar in bars_test:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')

            st.pyplot(fig)
        else:
            st.write("Metrics are not in the expected format for "
                     "visualization.")

        model_type = pipeline.model._type
        st.write(f"**Detected Model Type:** {model_type}")

        if model_type == "classification":
            actual = pipeline._test_y
            predicted = predictions

            cm_df = pd.crosstab(pd.Series(actual, name='Actual'),
                                pd.Series(predicted, name='Predicted'))

            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            cax = ax_cm.matshow(cm_df, cmap='Blues')
            fig_cm.colorbar(cax)
            ax_cm.set_xticks(range(len(cm_df.columns)))
            ax_cm.set_yticks(range(len(cm_df.index)))
            ax_cm.set_xticklabels([''] + list(cm_df.columns), rotation=45)
            ax_cm.set_yticklabels([''] + list(cm_df.index))
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('Actual')
            ax_cm.set_title('Confusion Matrix')

            for i in range(len(cm_df.index)):
                for j in range(len(cm_df.columns)):
                    ax_cm.text(j, i, cm_df.iloc[i, j], ha='center',
                               va='center', color='red')

            plt.tight_layout()
            st.pyplot(fig_cm)
            st.write("### Confusion Matrix:")
            st.dataframe(cm_df)

        elif model_type == "regression":
            try:
                actual = pipeline._test_y.flatten()
                predicted = predictions.flatten()

                st.write(f"**Actual Values Shape After Flattening:**"
                         f" {actual.shape}")
                st.write(f"**Predicted Values Shape After Flattening:**"
                         f" {predicted.shape}")

                residuals = actual - predicted

                fig_res, ax_res = plt.subplots(figsize=(10, 6))
                ax_res.scatter(predicted, residuals, alpha=0.5)
                ax_res.axhline(y=0, color='r', linestyle='--')
                ax_res.set_xlabel('Predicted Values')
                ax_res.set_ylabel('Residuals')
                ax_res.set_title('Residual Plot')
                plt.tight_layout()
                st.pyplot(fig_res)

                residual_mean = residuals.mean()
                residual_std = residuals.std()
                st.write(f"**Residuals Mean:** {residual_mean:.4f}")
                st.write(f"**Residuals Standard Deviation:**"
                         f" {residual_std:.4f}")
            except Exception as e:
                st.error(f"An error occurred while generating the residual"
                         f" plot: {e}")


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
        if 'results' not in st.session_state:
            results = pipeline.execute()
            st.session_state.results = results
            st.success("Pipeline executed")


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
    version = ask_for_input("Pipeline Version", "Version1.1")

    if st.button("Save Pipeline"):
        pipeline.execute()
        pipeline_artifact = pipeline.to_artifact(name=name, version=version)
        automl.registry.register(pipeline_artifact)
        st.success("Pipeline saved successfully!")
