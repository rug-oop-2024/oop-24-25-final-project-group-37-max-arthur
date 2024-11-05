import streamlit as st
import pandas as pd

from typing import Optional
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.artifact import Artifact


def choose_dataset(datasets: 'Dataset'):
    dataset_name = st.selectbox("Select a dataset", [dataset.name for dataset in datasets])
    artifact = next((dataset for dataset in datasets if dataset.name == dataset_name), None)
    st.write(f"### Selected Dataset: {artifact.name}")
    return artifact_to_dataset(artifact)


def display_datasets_accordion(automl, datasets):
    st.write("Available Datasets:")
    for artifact in datasets:
        dataset = artifact_to_dataset(artifact)
        with st.expander(dataset.name):
            st.dataframe(dataset.read().head())
            delete_dataset_button(automl, dataset, key=f"delete_{dataset.name}")


def slice_data_accordion(automl, datasets):
    st.write("Slice Dataset:")
    with st.expander("Slice Dataset"):
        dataset = choose_dataset(datasets)
        slice_criteria = st.text_input("Enter slice criteria (e.g., age > 18, smoker == 'yes', region in ['southeast', 'northwest']):")
        show_preview("Data Preview:", dataset.read())

        if slice_criteria:
            try:
                sliced_data = dataset.read().query(slice_criteria)
                show_preview("Sliced Data Preview:", sliced_data)
                name = ask_for_input("Enter sliced dataset name", f"sliced_{dataset.name}")
                asset_path = ask_for_input("Enter sliced asset path", f"sliced_{dataset.name}")
                # implement a check weather that name already exists

                save_df_to_dataset_button(automl, name, sliced_data, asset_path, key=f"save_{name}")
            except Exception as e:
                st.error(f"Error in slicing data: {e}")


def save_df_to_dataset_button(automl, name, data, asset_path, key=None):
    if st.button("Save Dataset"):
        dataset = Dataset.from_dataframe(name=name, data=data, asset_path=asset_path)
        automl.registry.register(dataset)
        st.success(f"Dataset '{name}' saved and registered successfully.")


def delete_dataset_button(automl, dataset, key=None):
    if st.button(f"Delete {dataset.name}"):
        automl.registry.delete(dataset.id)
        st.success(f"Dataset {dataset.name} deleted successfully.")


def artifact_to_dataset(artifact: 'Artifact'):
    return Dataset(name=artifact.name, data=artifact.data, asset_path=artifact.asset_path)


def upload_csv_button():
    uploaded_file = st.file_uploader("Create a Dataset from uploading a CSV file", type="csv")
    return uploaded_file


def ask_for_input(text: str, default: Optional[str] = None):
    return st.text_input(text, value=default)


def show_preview(text: str, data: pd.DataFrame):
    st.write(text)
    st.write(data.head())


def save_csv(automl, file):
    data = pd.read_csv(file)
    show_preview("Data Preview:", data)
    name = ask_for_input("Enter dataset name", file.name)
    asset_path = ask_for_input("Enter asset path", file.name)
    # implement a check weather that name already exists
    save_df_to_dataset_button(automl, name, data, asset_path,  key=f"save_{name}")
