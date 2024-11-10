import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

# MNIST-like format (28x28 grayscale)
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
])


def _load_and_preprocess_image(image_path: str) -> np.ndarray:
    """
    Load and preprocesses an image to a flattened array.

    Args:
        image_path (str): Path to the image file.

    Returns:
        np.ndarray: Flattened array of pixel values.
    """
    image = Image.open(image_path)
    image = transform(image)
    image_flattened = np.array(image).flatten()
    return image_flattened


def create_image_dataframe(image_paths: list[str]) -> pd.DataFrame:
    """
    Create a DataFrame from a list of image paths including labels.

    Note: This function expects the first character to indicate the
    integer label of the image.

    Args:
        image_paths (List[str]): List of file paths to images.

    Returns:
        pd.DataFrame: DataFrame where each row represents pixel values.
            If labels are provided, a 'label' column is added.

    Raises:
        AssertionError: If any first character of the filenames is not an
            integer.
    """
    try:
        labels = [int(path.split('/')[-1][0]) for path in image_paths]
    except ValueError:
        print("Error: The first character of the filename is not an integer.")

    data = [_load_and_preprocess_image(img_path) for img_path in image_paths]
    df = pd.DataFrame(data, columns=[
        f"{i + 1}x{j + 1}" for i in range(28) for j in range(28)
    ])

    df.insert(0, "label", labels)
    print(df)
    return df
