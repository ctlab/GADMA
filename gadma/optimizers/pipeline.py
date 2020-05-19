

class Pipeline(object):
    """
    Class for concatenation of different optimization and functions between.
    """
    def __init__(self, model):
        self.pipeline = list()
        self.opt_kwargs = list()
        self.transforms = list()

    def add_optimizer(self, optimizer_id, kwargs={}, transform=None):
        if optimizer_id in _registered_global_optimizers:
            self.pipeline.append(get_global_optimizer(optimizer_id))
        elif optimizer_id in _registered_local_optimizers:
            self.pipeline.append(get_global_optimizer(optimizer_id))
        self.opt_kwargs.append(kwargs)
        self.transforms.append(transform)

    def run(self, target, X_init=None, Y_init=None, callback=None):
        """
        Run pipeline for target.
        Target is any object with :func:`objective` function.
        """
        cur_target = target
        cur_X_init = X_init
        cur_Y_init = Y_init
        x_best_total = list()
        y_best_total = list()
        for optimizer, transform, kwargs in zip(self.pipeline, self.transforms, self.opt_kwargs):
            if isinstance(optimizer, GlobalOptimizer):
                kwargs['X_init'] = cur_X_init
                kwargs['Y_init'] = cur_Y_init
            elif isinstance(optimizer, LocalOptimizer):
                kwargs['x0'] = X_init
            x_best, y_best, X_total, Y_total = optimizer.optimize(cur_target.objective, cur_target.variables, **kwargs)
            x_best_total.append(x_best)
            y_best_total.append(y_best)
            if callback is not None:
                callback(cur_target, x_best_total, y_best_total)

            cur_target = transform(cur_target, x_best_total, y_best_total)
        return x_best_total, y_best_total
        
