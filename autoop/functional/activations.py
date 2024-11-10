from torch import Tensor, exp


def softmax(predictions: Tensor) -> Tensor:
    """
    Compute the softmax of the input tensor along the specified dimension.

    Args:
        predictions (Tensor): The input tensor containing prediction values.

    Returns:
        Tensor: A tensor where values represent the probabilities of each
            class, summing to 1 along the specified dimension.
    """
    exp_predictions = exp(predictions)
    return exp_predictions / exp_predictions.sum(dim=1, keepdim=True)


def sigmoid(predictions: Tensor) -> Tensor:
    """
    Compute the sigmoid of the input tensor.

    Args:
        predictions (Tensor): The input tensor containing prediction values.

    Returns:
        Tensor: A tensor with each value transformed by the sigmoid function,
            representing probabilities.
    """
    return 1 / (1 + exp(-predictions))
