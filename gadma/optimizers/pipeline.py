

class Pipeline(object):
    """
    Class for concatenation of different optimization and functions between.
    """
    def __init__(self):
        self.pipeline = list()
        self.opt_kwargs = list()
        self.transforms = list()

    def add_optimizer(self, optimizer_id, kwargs={}, transform=None):
        if optimizer_id in _registered_global_optimizers:
            self.pipeline.append(get_global_optimizer(optimizer_id))
        elif optimizer_id in _registered_local_optimizers:
            self.pipeline.append(get_global_optimizer(optimizer_id))
        else:
            raise ValueError(f"Optimizer {optimizer_id} is not registered.")
        self.opt_kwargs.append(kwargs)
        self.transforms.append(transform)

    def run(self, target, X_init=None, Y_init=None, callback=None):
        """
        Run pipeline for target.
        Target is any object with :func:`evaluate` function.
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
                kwargs['x0'] = cur_X_init
            x_best, y_best, X_total, Y_total = optimizer.optimize(cur_target.evaluate, cur_target.variables, **kwargs)
            x_best_total.append(x_best)
            y_best_total.append(y_best)
            if callback is not None:
                callback(cur_target, x_best_total, y_best_total)

            cur_target, cur_X_init, cur_Y_init = transform(cur_target, cur_X_init, cur_Y_init)
        return x_best_total, y_best_total


class AutoPipeline(Pipeline):
    def add_optimizer(self, optimizer_id, kwargs={}):
        if len(self.pipeline) > 0:
            if isinstance(self.pipeline[-1], GlobalOptimizer):
                prev_is_global = True
            else:
                prev_is_global = False

            if optimizer_id in _registered_global_optimizers:
                cur_is_global = True
            elif optimizer_id in _registered_local_optimizers:
                cur_is_global = False
            else:
                raise ValueError(f"Optimizer {optimizer_id} is not registered.")

            if prev_is_global and not cur_is_global:
                self.transform[-1] = Target.to_local_optimizer
            if prev_is_local
                
