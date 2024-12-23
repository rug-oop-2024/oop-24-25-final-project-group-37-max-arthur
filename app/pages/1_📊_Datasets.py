import streamlit as st

import app.core.dataset_handler as dh
from app.core.system import AutoMLSystem


def render_datasets() -> None:
    """
    Renders the datasets page in the Streamlit app.
    This function sets up the page configuration, retrieves the list of
    datasets from the AutoML system, and displays them using various helper
    functions. It also provides functionality to upload and save new datasets.

    Args:
        None
    Returns:
        None
    """

    st.set_page_config(page_title="Datasets", page_icon="📊")

    automl = AutoMLSystem.get_instance()

    datasets = automl.registry.list(type="dataset")

    if datasets:
        dh.display_datasets_accordion(automl, datasets)

        dh.slice_data_accordion(automl, datasets)

    else:
        st.write("No datasets found.")

    file = dh.upload_csv_button()

    if file is not None:
        dh.save_csv(automl, file)

    dh.upload_image_button(automl)


if __name__ == "__main__":
    render_datasets()
