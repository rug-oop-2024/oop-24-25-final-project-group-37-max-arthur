import streamlit as st
import app.core.modelling_handler as mh
from app.core.system import AutoMLSystem
from app.core.dataset_handler import choose_dataset
from autoop.core.ml.pipeline import Pipeline


def render_modelling():
    st.set_page_config(page_title="Modelling", page_icon="📈")

    st.write("# ⚙ Modelling")

    mh.write_helper_text("In this section, you can design a machine learning"
                         " pipeline to train a model on a dataset.")

    automl = AutoMLSystem.get_instance()

    datasets = automl.registry.list(type="dataset")

    selected_dataset = choose_dataset(datasets)

    selected_target_column = mh.choose_target_column(selected_dataset)

    selected_input_columns = mh.choose_input_columns(
        selected_dataset, selected_target_column
        )

    target_feature, input_features = mh.generate_target_and_input_features(
        selected_dataset, selected_target_column, selected_input_columns
        )

    model_type = mh.determine_task_type(target_feature)
    st.write(f"Model type determined: {model_type}")

    selected_model = mh.choose_model(model_type)

    dataset_split = st.slider("Select the dataset split", 0.1, 0.9, 0.8, 0.1)

    selected_metrics = mh.choose_metrics(model_type)

    required_elements = [
        selected_dataset, selected_metrics, target_feature,
        input_features, selected_model, dataset_split
    ]

    if all(required_elements):

        pipeline = Pipeline(
            dataset=selected_dataset,
            metrics=selected_metrics,
            target_feature=target_feature,
            input_features=input_features,
            model=selected_model,
            split=dataset_split
            )

        mh.display_pipeline_summary(pipeline)

        mh.execute_pipeline_button(pipeline)

        mh.save_pipeline_button(automl, pipeline)


if __name__ == "__main__":
    render_modelling()
