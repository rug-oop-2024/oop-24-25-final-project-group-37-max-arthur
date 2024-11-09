import pickle
import streamlit as st
import numpy as np
import app.core.dataset_handler as dh

from app.core.system import AutoMLSystem
from autoop.core.ml.artifact import Artifact
from autoop.core.ml.pipeline import Pipeline
from autoop.core.ml.model import Model
from autoop.functional.feature import detect_feature_types
from autoop.functional.preprocessing import preprocess_features

from typing import List, Tuple


def delete_pipeline_button(automl: 'AutoMLSystem', name: str, id: str) -> None:
    """
    Displays a button to delete a pipeline and handles the deletion process.
    Args:
        automl (AutoMLSystem): The AutoML system instance that manages the
        pipelines.
        name (str): The name of the pipeline to be deleted.
        id (str): The unique identifier of the pipeline to be deleted.
    Returns:
        None
    """

    if st.button(f"Delete {name}"):
        automl.registry.delete(id)
        st.success(f"Pipeline {name} was deleted successfully.")


def select_pipeline(pipelines: List['Artifact']) -> Tuple['Pipeline', str,
                                                          'Artifact']:
    """
    Selects a pipeline from a list of pipeline artifacts.
    Args:
        pipelines (List['Artifact']): A list of pipeline artifacts to choose
        from.
    Returns:
        Tuple['Pipeline', str, 'Artifact']: A tuple containing the selected
        pipeline, the name of the selected pipeline, and the selected pipeline
        artifact.
    """

    name = st.selectbox(
        "Select a pipeline",
        [pipeline_artifact.name for pipeline_artifact in pipelines]
                        )

    selected_pipeline_artifact = next(
        (p for p in pipelines if p.name == name), None
        )
    pipeline = pickle.loads(selected_pipeline_artifact.read())
    return pipeline, name, selected_pipeline_artifact


def choose_data(automl: 'AutoMLSystem') -> 'Artifact':
    """
    Prompts the user to select a CSV file or choose an existing dataset
    for making predictions.
    Args:
        automl (AutoMLSystem): The AutoML system instance used for managing
        datasets.
    Returns:
        Artifact: The selected dataset artifact.
    """
    st.write("### Select Data")
    st.write("Select a CSV file or choose a dataset to make predictions.")
    file = dh.upload_csv_button()
    if file is not None:
        dh.save_csv(automl, file)
        st.write("Refresh to choose just uploaded CSV file.")
    datasets = automl.registry.list(type="dataset")
    return dh.choose_dataset(datasets)


def predict_button(compact_observation_vector: np.ndarray,
                   pipeline: 'Pipeline', model: 'Model') -> None:
    """
    Handles the prediction button click event in a Streamlit application.
    Args:
        compact_observation_vector (np.ndarray): The input data for making
        predictions.
        pipeline (Pipeline): The pipeline object containing the model and its
        parameters.
        model (Model): The model object used for making predictions.
    Returns:
        None
    """
    st.write(pipeline.model.parameters)
    if st.button("Predict"):
        predictions = model.predict(compact_observation_vector)
        st.write("### Predictions")
        st.write(predictions)


def preprocess_data(dataset: 'Artifact', pipeline: 'Pipeline') -> np.ndarray:
    """
    Preprocesses the data by detecting feature types, validating pipeline
    input features, and concatenating input vectors into a compact observation
    vector.
    Args:
        dataset (Artifact): The dataset to be preprocessed.
        pipeline (Pipeline): The pipeline containing the input features and
        execution logic.
    Returns:
        np.ndarray: The compact observation vector after preprocessing.
    Raises:
        ValueError: If the input features of the pipeline are not present in
        the new data.
    """

    feature_types = detect_feature_types(dataset)

    pipeline_input_features = pipeline._input_features
    for feature in pipeline_input_features:
        if str(feature) not in [str(f) for f in feature_types]:
            st.error("Input features of the pipeline are not present in "
                     "the new data. Please choose a different dataset.")
            raise ValueError(
                "Input features of the pipeline are not present in the new "
                "data. Please choose a different dataset.")
            return

    input_results = preprocess_features(pipeline_input_features, dataset)
    input_vectors = [data for (feature_name, data, artifact) in input_results]
    compact_observation_vector = np.concatenate(input_vectors, axis=1)
    pipeline.execute()
    return compact_observation_vector


def prediction_accordion(automl: 'AutoMLSystem',
                         pipeline: 'Pipeline', model: 'Model') -> None:
    """
    Displays an interactive accordion for making predictions using a given
    AutoML system, pipeline, and model.
    Args:
        automl (AutoMLSystem): The automated machine learning system to use
        for selecting the dataset.
        pipeline (Pipeline): The data processing pipeline.
        model (Model): The machine learning model to use for making
        predictions.
    Returns:
        None
    """

    with st.expander("### Predict", expanded=True):
        dataset = choose_data(automl)

        if dataset is not None:
            st.write("### Data Preview")
            st.write(dataset.read().head())
            observations = preprocess_data(dataset, pipeline)
            predict_button(observations, pipeline, model)
