import numpy as np

from edugrad.ops import Operation, tensor_op


@tensor_op
class sigmoid(Operation):
    @staticmethod
    def forward(ctx, a):
        out = 1 / (1 + np.exp(-a))
        ctx.append(out)  # Save sigmoid result for use in backward
        return out
        

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_out = ctx[0]
        grad_input = grad_output * sigmoid_out * (1 - sigmoid_out)
        return [grad_input]


@tensor_op
class log(Operation):
    @staticmethod
    def forward(ctx, a):
        ctx.append(a)
        return np.log(a)
        

    @staticmethod
    def backward(ctx, grad_output):
        a = ctx[0]
        grad_input = grad_output / a
        return [grad_input]


@tensor_op
class multiply(Operation):
    """Element-wise multiplication. """

    @staticmethod
    def forward(ctx, a, b):
        ctx.append((a, b))
        return a * b

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx[0]
        grad_a = grad_output * b
        grad_b = grad_output * a
        return [grad_a, grad_b]
    
@tensor_op
class add(Operation):
    """Element-wise addition: a + b"""

    @staticmethod
    def forward(ctx, a, b):
        return a + b

    @staticmethod
    def backward(ctx, grad_output):
        return [grad_output, grad_output]

    
@tensor_op
class subtract(Operation):
    """Element-wise subtraction: a - b"""

    @staticmethod
    def forward(ctx, a, b):
        return a - b

    @staticmethod
    def backward(ctx, grad_output):
        return [grad_output, -grad_output]



@tensor_op
class sum_along_columns(Operation):
    @staticmethod
    def forward(ctx, a):
        ctx.append(a)
        return np.sum(a, axis=1)

    @staticmethod
    def backward(ctx, grad_output):
        return [np.ones(ctx[-1].shape) * grad_output[:, np.newaxis]]


@tensor_op
class lookup_rows(Operation):
    """Given a matrix of size [m, n] and an array of integer indices
    [i0, i1, ..., in], return the relevant rows of the matrix.
    """

    @staticmethod
    def forward(ctx, matrix, indices):
        ctx.append(matrix.shape)
        ctx.append(indices)
        return matrix[indices]

    @staticmethod
    def backward(ctx, grad_output):
        shape, indices = ctx
        grads = np.zeros(shape)
        # this is some numpy magic: `indices` may have repeats of a given token index,
        # but if we just do grads[indices] += grad_output, it won't add up the rows
        # from grad_output for each occurance of the same index; this method accumulates
        # all of those sums, which is what's needed for the gradients
        np.add.at(grads, indices, grad_output)
        return [grads, np.zeros(indices.shape)]
    
@tensor_op
class reduce_mean(Operation):
    """Computes the mean of all elements in the tensor."""

    @staticmethod
    def forward(ctx, a):
        ctx.append(a.shape)
        return np.array(np.mean(a))

    @staticmethod
    def backward(ctx, grad_output):
        shape = ctx[0]
        size = np.prod(shape)
        return [np.ones(shape) * grad_output / size]

