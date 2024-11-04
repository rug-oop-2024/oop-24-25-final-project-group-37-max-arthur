import streamlit as st
# import pandas as pd

from app.core.system import AutoMLSystem
# from autoop.core.ml.dataset import Dataset

from app.core.dataset_handler import upload_csv_button, save_csv, display_datasets_accordion, slice_data_accordion


def render_datasets():
    st.set_page_config(page_title="Datasets", page_icon="📊")

    automl = AutoMLSystem.get_instance()

    datasets = automl.registry.list(type="dataset")

    display_datasets_accordion(automl, datasets)

    slice_data_accordion(automl, datasets)

    file = upload_csv_button()

    if file is not None:
        save_csv(automl, file)


if __name__ == "__main__":
    render_datasets()
