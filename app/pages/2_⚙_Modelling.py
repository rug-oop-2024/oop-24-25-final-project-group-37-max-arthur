import streamlit as st

from app.core.system import AutoMLSystem
from app.core.modelling_handler import choose_metrics, determine_task_type, choose_target_column, choose_model, generate_target_and_input_features, choose_input_columns, display_pipe_line_summary, write_helper_text, execute_pipeline_button
from app.core.dataset_handler import choose_dataset


def render_modelling():
    st.set_page_config(page_title="Modelling", page_icon="📈")

    st.write("# ⚙ Modelling")

    write_helper_text("In this section, you can design a machine learning pipeline to train a model on a dataset.")

    automl = AutoMLSystem.get_instance()

    datasets = automl.registry.list(type="dataset")

    selected_dataset = choose_dataset(datasets)

    selected_target_column = choose_target_column(selected_dataset)

    selected_input_columns = choose_input_columns(selected_dataset, selected_target_column)

    target_feature, input_features = generate_target_and_input_features(selected_dataset, selected_target_column, selected_input_columns)

    model_type = determine_task_type(target_feature)
    st.write(f"Model type determined: {model_type}")

    selected_model = choose_model(model_type)

    dataset_split = st.slider("Select the dataset split", 0.1, 0.9, 0.8, 0.1)

    # Prompt the user to select a set of compatible metrics. Should we determine based on which model type which metrics are most useful?
    selected_metrics, metrics_names = choose_metrics()

    display_pipe_line_summary(selected_dataset, selected_target_column, selected_input_columns, model_type, selected_model, metrics_names, dataset_split)

    if selected_dataset and selected_metrics and target_feature and input_features and selected_model and dataset_split:
        execute_pipeline_button(selected_dataset, selected_metrics, target_feature, input_features, selected_model, dataset_split)


if __name__ == "__main__":
    render_modelling()
