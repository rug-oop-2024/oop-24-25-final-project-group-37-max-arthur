# Instructions 📝

Welcome to the AutoML Streamlit app! This guide provides an overview of each main page—**Datasets**, **Modelling**, and **Deployment**—and instructions on how to use them.

## 1. Datasets Page 📊

The Datasets page allows you to manage datasets within the app.

- **View Datasets**: Browse the available datasets in the system. Each dataset can be expanded for detailed insights.
- **Slice Data**: Use the slicing options to create subsets of your dataset for more targeted analyses.
- **Upload New Dataset**: You can upload a CSV file using the upload button. Once uploaded, the dataset can be saved in the system for future use.
- **Upload Images**: You can also upload images to create training - or inference datasets.

### How to Use:
1. Choose a dataset to view and explore its details.
2. (Optional) Use the “Slice Data” option to filter data by specific features and save the sliced dataset.
3. If you have your own dataset or image data, upload it via the "Upload CSV" button and save it.

## 2. Modelling Page ⚙

On the Modelling page, you can configure and run machine learning models on your dataset.

- **Select Dataset**: Start by selecting a dataset for training.
- **Choose Target and Input Columns**: Specify the target column (what you want to predict) and input columns (features for prediction).
- **Configure Model and Training**:
  - The system automatically determines the model type (e.g., classification, regression) based on the target column.
  - In case of an integer target column, you have the option to have the system view it as categorical and use classification.
  - Select a model suited to your task.
  - Adjust the dataset split for training and testing.
  - Choose evaluation metrics based on your model type.
- **Execute Pipeline**: Once configured, execute the pipeline to train your model.
- **View results**: View the _Pipeline Summary_ and the results of the specified metrics.
- **Save Pipeline**: Save the pipeline and model configuration. You can use the model in production, and can use the pipeline to predict under **Deployment**.

### How to Use:
1. Select a dataset and configure target/input columns.
2. Choose a model and configure any relevant parameters.
3. Execute the pipeline to train the model and view the results.
4. If satisfied with the model, save the pipeline for deployment.

## 3. Deployment Page 💈

The Deployment page lets you manage saved pipelines and make predictions.

- **Select a Pipeline**: Choose a previously saved pipeline from the list to deploy.
- **Pipeline Management**:
  - View a summary of the selected pipeline, including model parameters and metrics.
  - Delete a pipeline if it’s no longer needed.
- **Make Predictions**: Use the selected pipeline to make predictions on new data inputs. Predictions will display directly within the app and can be downloaded.

### How to Use:
1. Select a pipeline from the list to view details or make predictions.
2. Enter data in the prediction fields to test the model's predictions.
3. You can also delete pipelines from this page if necessary.

---

**Enjoy!**

For more info, refer to the docs in our [GitHub repo](https://github.com/rug-oop-2024/oop-24-25-final-project-group-37-max-arthur).
