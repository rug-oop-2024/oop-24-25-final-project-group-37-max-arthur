import pickle
import streamlit as st
import numpy as np
import app.core.dataset_handler as dh
import pandas as pd
import base64

from torch import Tensor
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
    st.write("Upload a CSV file to create da dataset or/and choose a dataset"
             " to make predictions.")
    file = dh.upload_csv_button()
    if file is not None:
        dh.save_csv(automl, file)
        st.write("Refresh to choose just uploaded CSV file.")
    datasets = automl.registry.list(type="dataset")
    return dh.choose_dataset(datasets)


def download_df(dataframe: pd.DataFrame, filename: str, linktext: str) -> None:
    """
    Generates a download link for a given DataFrame and displays it in a
    Streamlit app.
    Args:
        dataframe (pd.DataFrame): The DataFrame to be downloaded.
        filename (str): The name of the file to be downloaded.
        linktext (str): The text to be displayed for the download link.
    Returns:
        None
    """
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    string = "data:file/csv;base64"
    href = f'<a href="{string},{b64}" download="{filename}">{linktext}</a>'
    st.markdown(href, unsafe_allow_html=True)


def download_pickled_model(model: 'Model') -> None:
    """
    Serializes a given model using pickle and provides a Streamlit download
    button for the pickled model.
    Args:
        model (Model): The model to be serialized and downloaded.
    Returns:
        None
    """

    data = pickle.dumps(model)
    st.download_button(
        label="Download Pickled Model",
        data=data,
        file_name=f'{model.__class__.__name__}.pkl',
        mime='application/octet-stream'
    )


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
    st.write("### Model Parameters")
    st.write(pipeline.model.parameters)
    download_pickled_model(model)

    if st.button("Predict"):
        predictions = model.predict(compact_observation_vector)
        st.write("### Predictions")
        show_predictions(predictions, pipeline, compact_observation_vector)

        if isinstance(predictions, Tensor):
            df = pd.DataFrame(predictions.numpy())
            download_df(df, "predictions.csv", "Download Predictions CSV")


def get_feature_names(pipeline: 'Pipeline') -> List[str]:
    """
    Extracts the feature names from the pipeline's input features.
    Args:
        pipeline (Pipeline): The pipeline object containing the input features.
    Returns:
        List[str]: A list of feature names.
    """
    feature_names = []
    for feature in pipeline._input_features:
        if feature.type == "categorical":
            feature_names.extend(
                [f"{feature.name}_class_{i}" for i in range(
                    feature.num_options)])
        else:
            feature_names.append(feature.name)
    return feature_names


def show_predictions(predictions: Tensor, pipeline: 'Pipeline',
                     observation_vector: np.ndarray) -> None:
    with st.expander("Show Predictions"):
        if isinstance(predictions, Tensor):
            predictions = predictions.numpy()
        feature_names = get_feature_names(pipeline)

        data = observation_vector

        df = pd.DataFrame(data, columns=feature_names)
        df['Predictions'] = predictions

        st.write(df)


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

    input_results = preprocess_features(pipeline_input_features, dataset)
    input_vectors = [data for (feature_name, data, artifact) in input_results]
    compact_observation_vector = np.concatenate(input_vectors, axis=1)
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

    with st.container(border=True):
        dataset = choose_data(automl)
        if dataset is not None:
            st.write("### Data Preview")
            st.write(dataset.read().head())

    with st.container(border=True):
        st.write("### Predict")
        if dataset is not None:
            observations = preprocess_data(dataset, pipeline)
            predict_button(observations, pipeline, model)
