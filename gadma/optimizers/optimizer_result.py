import scipy

class OptimizerResult(object):
    """
    Class for keeping optimizers result.
    It is based on SciPy.optimize.OptimizeResult but have more information.

    :param x: The solution of the optimization. The best value during run.
    :param y: The value of objective function on x.
    :param success: Whether or not the optimizer exited successfully.
    :type success: bool
    :param status: Termination status of the optimizer. Its value depends on
                   the underlying solver. Refer to message for details.
    :type status: int
    :param message: Description of the cause of the termination.
    :type message: str
    :param X: All solutions that were used in run.
    :param Y: Values of objective function on X.
    :param n_eval: Number of evaluations of the objective functions performed
                   by the optimizer.
    :type n_eval: int
    :param n_iter: Number of iterations performed by the optimizer.
    :type n_iter: int
    """
    def __init__(self, x, y, success: bool, status: int, message: str, X, Y, n_eval: int, n_iter:int):
        self.x = x
        self.y = y
        self.success = success
        self.status = status
        self.message = message
        self.X = X
        self.Y = Y
        self.n_eval = n_eval
        self.n_iter = n_iter

    @staticmethod
    def from_SciPy_OptimizeResult(scipy_result: scipy.optimize.OptimizeResult):
        """
        Create OptimizerResult from instance of SciPy.optimize.OptimizeResult.
        Please, note that some attributes will be empty.
        """
        return OptimizerResult(x=scipy_result.x, y=scipy_result.fun,
                               success=scipy_result.success,
                               status=scipy_result.status,
                               message=scipy_result.message,
                               X=[], Y=[], n_eval=scipy_result.nfev,
                               n_iter=scipy_result.nit)
