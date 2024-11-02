from torch import Tensor, exp

def softmax(predictions: Tensor) -> Tensor:
    exp_predictions = exp(predictions)
    return exp_predictions / exp_predictions.sum(dim=1, keepdim=True)

def sigmoid(predictions: Tensor) -> Tensor:
    return 1 / (1 + exp(-predictions))
