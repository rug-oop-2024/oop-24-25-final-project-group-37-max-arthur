import streamlit as st
import pandas as pd

from typing import Optional, List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact
from app.core.system import AutoMLSystem


def choose_dataset(datasets: List['Artifact']) -> 'Dataset':
    """
    Prompts the user to select a dataset from a list of datasets and returns
    the selected dataset.

    Args:
        datasets (List['Artifact']): A list of Artifact objects representing
        available datasets.

    Returns:
        Dataset: The selected dataset converted from the chosen Artifact.
    """
    dataset_name = st.selectbox("Select a dataset",
                                [dataset.name for dataset in datasets])
    artifact = next(
        (dataset for dataset in datasets if dataset.name == dataset_name),
        None
        )
    # st.write(f"### Selected Dataset: {artifact.name}")
    return artifact_to_dataset(artifact)


def display_datasets_accordion(automl: 'AutoMLSystem',
                               datasets: List['Artifact']) -> None:
    """
    Displays an accordion of datasets using Streamlit.
    Args:
        automl (AutoMLSystem): The AutoML system instance.
        datasets (List[Artifact]): A list of dataset artifacts.
    Returns:
        None
    """
    st.write("Available Datasets:")
    for artifact in datasets:
        dataset = artifact_to_dataset(artifact)
        with st.expander(dataset.name):
            st.dataframe(dataset.read().head())
            delete_dataset_button(automl, dataset)


def slice_data_accordion(automl: 'AutoMLSystem', datasets: List['Artifact']
                         ) -> None:
    """
    Displays a Streamlit interface for slicing a dataset based on user-defined
    criteria.

    Args:
        automl (AutoMLSystem): The AutoML system instance.
        datasets (List[Artifact]): A list of dataset artifacts to choose from.

    Raises:
        Exception: If there is an error in slicing the data based on the
        provided criteria.
    """

    st.write("Slice Dataset:")
    with st.expander("Slice Dataset"):
        dataset = choose_dataset(datasets)
        slice_criteria = st.text_input(
            "Enter slice criteria (e.g., age > 18,"
            "smoker == 'yes', region in ['southeast', 'northwest']):")
        show_preview("Data Preview:", dataset.read())

        if slice_criteria:
            try:
                sliced_data = dataset.read().query(slice_criteria)
                show_preview("Sliced Data Preview:", sliced_data)
                name = ask_for_input("Enter sliced dataset name",
                                     f"sliced_{dataset.name}")
                asset_path = ask_for_input("Enter sliced asset path",
                                           f"sliced_{dataset.name}")
                # implement a check weather that name already exists

                save_df_to_dataset_button(automl, name, sliced_data,
                                          asset_path)
            except Exception as e:
                st.error(f"Error in slicing data: {e}")


def save_df_to_dataset_button(automl: 'AutoMLSystem', name: str,
                              data: pd.DataFrame, asset_path: str
                              ) -> None:
    """
    Displays a button in the Streamlit app to save a DataFrame as a dataset
    and register it with the AutoML system.
    Args:
        automl (AutoMLSystem): The AutoML system instance to register
        the dataset with.
        name (str): The name of the dataset.
        data (pd.DataFrame): The DataFrame containing the dataset.
        asset_path (str): The file path where the dataset asset will be saved.
    Returns:
        None
    """
    if st.button("Save Dataset"):
        dataset = Dataset.from_dataframe(name=name, data=data,
                                         asset_path=asset_path)
        automl.registry.register(dataset)
        st.success(f"Dataset '{name}' saved and registered successfully.")


def delete_dataset_button(automl: 'AutoMLSystem', dataset: List['Artifact']
                          ) -> None:
    """
    Handles the deletion of a dataset through a Streamlit button.
    Args:
        automl (AutoMLSystem): The AutoML system instance that manages datasets
        dataset (List[Artifact]): The dataset to be deleted.
    Returns:
        None
    """
    if st.button(f"Delete {dataset.name}"):
        automl.registry.delete(dataset.id)
        st.success(f"Dataset {dataset.name} deleted successfully.")


def artifact_to_dataset(artifact: 'Artifact') -> 'Dataset':
    """
    Converts an Artifact object to a Dataset object.
    Args:
        artifact (Artifact): The Artifact object to be converted.
    Returns:
        Dataset: A new Dataset object created from the given Artifact.
    """
    return Dataset(name=artifact.name, data=artifact.data,
                   asset_path=artifact.asset_path)


def upload_csv_button() -> 'st.runtime.uploaded_file_manager.UploadedFile':
    """
    Displays a file uploader widget for CSV files and returns the uploaded
    file.
    Returns:
        Optional[st.uploaded_file_manager.UploadedFile]: The uploaded CSV
        file.
    """
    uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
    return uploaded_file


def ask_for_input(text: str, default: Optional[str] = None) -> str:
    """
    Displays a text input widget and returns the user's input.
    Args:
        text (str): The label for the text input widget.
        default (Optional[str], optional): The default value for the text
        input. Defaults to None.
    Returns:
        str: The user's input as a string.
    """
    return st.text_input(text, value=default)


def show_preview(text: str, data: pd.DataFrame) -> None:
    """
    Displays a preview of the given DataFrame along with a text description.
    Args:
        text (str): The text description to display.
        data (pd.DataFrame): The DataFrame to preview.
    Returns:
        None
    """
    st.write(text)
    st.write(data.head())


def save_csv(automl: 'AutoMLSystem',
             file: 'st.runtime.uploaded_file_manager.UploadedFile') -> None:
    """
    This function reads the uploaded CSV file, shows a preview of the data,
    and prompts the user to enter a dataset name and asset path.
    It then saves the dataset to the AutoML system using the provided name and
    asset path.    Args:
        automl (AutoMLSystem): The AutoML system instance where the dataset
        will be saved.
        file (st.uploaded_file_manager.UploadedFile): The uploaded CSV file to
        be saved.
    Returns:
        None
    """

    data = pd.read_csv(file)
    show_preview("Data Preview:", data)
    name = ask_for_input("Enter dataset name", file.name)
    asset_path = ask_for_input("Enter asset path", file.name)
    # implement a check weather that name already exists
    save_df_to_dataset_button(automl, name, data, asset_path)
