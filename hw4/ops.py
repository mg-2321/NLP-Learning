import numpy as np
from edugrad.ops import Operation, tensor_op, reduce_mean, relu
from edugrad.tensor import Tensor

from hw3_ops import sum_along_columns, log, multiply


@tensor_op
class divide(Operation):
    """Divide row-wise a [batch_size, dimension] Tensor by a [batch-size]
    Tensor of scalars.

    Example:
        a = Tensor(np.array([[1., 2.], [3., 4.]]))
        b = Tensor(np.array([2., 3.]))
        divide(a, b).value == np.array([[0.5, 1.0], [1.0, 1.33]])
    """

    @staticmethod
    def forward(ctx, a, b):
        ctx.append(a)
        ctx.append(b)
        # broadcast b to [batch_size, 1]
        return a / b[:, np.newaxis]

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx[-2:]
        # broadcast b to a column
        b_column = b[:, np.newaxis]
        # a gradients
        inv_b_column = 1 / b_column
        num_columns = a.shape[1]
        broadcast = np.hstack([inv_b_column] * num_columns)
        a_grads = broadcast * grad_output
        # d(a/b) / dx for each x in a
        squared_inv = -a * b_column ** -2
        multiply_upstream = squared_inv * grad_output
        # sum along paths to get d(a/b) / db
        b_local_grads = np.sum(multiply_upstream, axis=1)
        return a_grads, b_local_grads


@tensor_op
class exp(Operation):
    """ e^x element-wise """

    @staticmethod
    def forward(ctx, a):
        value = np.exp(a)
        ctx.append(value)
        return value

    @staticmethod
    def backward(ctx, grad_output):
        value = ctx[-1]
        return [value * grad_output]


def softmax_rows(logits: Tensor) -> Tensor:
    """Compute softmax of a batch of inputs.
    e^x / sum(e^x), where the sum is taken per-row

    Args:
        logits: [batch_size, num_classes] containing logits

    Returns: [batch_size, num_classes] Tensor
        row-wise softmax of logits, i.e. each row will be a probability distribution.
    """
    # TODO (~3 lines): implement here
    # HINT: sum_along_columns is a friend
    exps = exp(logits)
    row_sums = sum_along_columns(exps)
    return divide(exps, row_sums)


def cross_entropy_loss(probabilities: Tensor, labels: Tensor) -> Tensor:
    """Compute mean cross entropy.

    Args:
        probabilities: [batch_size, num_labels], each row is a probability distribution
        labels: [batch_size, num_labels], each row is a probability distribution
            (typically, each row will be a one-hot)

    Returns:
        1 / batch_size * sum_row cross_entropy(labels[row], probabilities[row])
        where cross_entropy(p, q) = - sum(p[i] * log(q[i]))
    """
    # TODO (~5 lines): implement
    # you can use items available from hw3_ops, ones defined above, as well as reduce_mean,
    # which has been imported from edugrad
    # helper scalar
    """negative_one = Tensor(np.array(-1.0))
    log_probs = log(probabilities)
    product = multiply(labels, log_probs)
    summed = sum_along_columns(product)
    mean_loss = reduce_mean(summed)
    neg_loss = multiply(summed, negative_one)
    return (neg_loss)
"""
    negative_one = Tensor(np.array(-1.0))  # helper scalar
    log_probs = log(probabilities)                     # log(q)
    loss_per_example = multiply(labels, log_probs)     # p * log(q)
    summed_loss = sum_along_columns(loss_per_example)  # sum over classes
    mean_loss = reduce_mean(summed_loss)               # mean over batch
    #print(multiply(negative_one, mean_loss) ) 
    return multiply(negative_one, mean_loss)           # final: -mean