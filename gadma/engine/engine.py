from ..code_generator import id2printfunc

_registered_engines = {}

def register_engine(engine):
    """
    Registers the specified engine of the demographic inference.

    :raises ValueError: if engine with the same `id` was already registered. 
    """
    if engine.id in _registered_engines:
        raise ValueError(f"Engine of the demographic inference '{engine.id}' already registered.")
    _registered_engines[engine.id] = engine


def get_engine(id):
    """
    Returns the engine of the demographic inference with the specified id.

    :raises ValueError: if engine with such `id` was not registered.
    """
    if id not in _registered_engines:
        raise ValueError(f"Engine of the demographic inference '{id}' not registered")
    return _registered_engines[id]


def all_engines():
    """
    Returns an iterator over all registered engines of the demographic inference.
    """
    for engine in _registered_engines.values():
        yield engine


class Engine(object):
    """
    Abstract class representing an engine of the demographic inference.

    New engine should be inheritted from this class.
    Engine must have at least the ``id``, ``supported_event_classes``,
    ``supported_data_classes`` and ``data_type`` attributes, and 
    implementations of :func:`read_data`, :func:`get_pop_labels`, 
    :func:`get_sample_sizes`, :func:`get_outgroup`, :func:`get_seq_len`
    and :func:`objective_function` functions of this abstract class.

    :cvar str id: the unique identifier of the engine.
    :cvar supported_event_classes: list of supported :class:`Event` classes.
    :cvar supported_data_classes: list of supported :class:`DataHolder` classes.
    :cvar data_type: class of data that is returned from :func:`read_data`
    """
    supported_event_classes = []
    supported_data_classes = []
    data_type = None

    @staticmethod
    def read_data(data_holder):
        """
        Reads data from `data_holder.filename` in inner type.

        :param data_holder: Holder of data to read.
        :type data_holder: :class:`gadma.DataHolder`

        :returns: readed data
        :rtype: ``Engine.data_type``
        """
        raise NotImplementedError

    @staticmethod
    def get_pop_labels(data):
        """
        Returns population labels from the data.

        :param data: data to process.
        :type: inner type
        """
        raise NotImplementedError

    @staticmethod
    def get_sample_sizes(data):
        """
        Returns populations sample sizes of the data.

        :param data: data to process.
        :type: inner type
        """
        raise NotImplementedError

    @staticmethod
    def get_outgroup(data):
        """
        Returns if there is an outgroup in the data.

        :param data: data to process.
        :type: inner type
        """
        raise NotImplementedError

    @staticmethod
    def get_seq_len(data):
        """
        Returns length of sequence that was used to build the data.

        :param data: data to process.
        :type: inner type
        """
        raise NotImplementedError

    def set_demographic_model(self, demographic_model):
        """
        Sets new demographic model for the engine.

        :param demographic_model: new demographic model.
        :type demographic_model: :class:`DemographicModel`

        :raises ValueError: when demographic model has events that are not\
            supported by the engine.
        """
        for event in demographic_model:
            if event.__class__ not in self.supported_events_classes:
                raise ValueError(f"Demographic model has event(s) that are not supported for {self.id} engine.\nThe supported event classes are: {self.supported_event_classes}")  
        self.demographic_model = demographic_model

    def set_data(self, data_holder):
        """
        Sets new data for the engine.

        :param data_holder: new data.
        :type data_holder: :class:`DataHolder`.

        :raises ValueError: when `data_holder` is not supported by the engine.
        """
        if data_holder.__class__ not in self.supported_data_classes:
            raise Exception(f"Data class {data_holder.__class__.__name__} is not in supported classes of {self.id} engine.\nThe supported classes are: {self.supported_data_classes}")
        if self.id != data_holder.ready_for_engine:
            data_holder.prepare_for_engine(self)
        self.data_holder = data_holder

    def objective_function(self, values):
        """
        Evaluation of the objective function of the engine.

        :param values: values of variables of setted demographic model.
        """
        raise NotImplementedError

    def evaluate(self, values, demographic_model=None, data_holder=None):
        """
        Sets demographic model and data holder for engine instance to 
        `demographic_model` and `data_holder` (if they are not None).
        And evaluates the :func:`objective_function`. 

        :param values: values of variables of demographic model.
        :param demographic_model: demographic model.
        :type demographic_model: :class:`gadma.DemographicModel`
        :param data_holder: holder of the data
        :type data_holder: :class:`gadma.DataHolder`

        :raises ValueError: if `demographic_model` is None and any was not setted\
            before; or if `data_holder` is None and any was not setted before.
        """
        if demographic_model is not None:
            self.set_demographic_model(demographic_model)
        if data_holder is not None:
            self.set_data(data_holder)
        if self.demographic_model is None:
            raise ValueError("Please set demographic model for engine or pass it as argument to evaluate function.")
        if self.data_holder is None:
            raise ValueError("Please set data for engine or pass it as argument to evaluate function.")

        return self.objective_function(values)

    def print_to_file(self, values, filename):
        """
        Prints nice formated code in the format of engine to file.
        """
        id2printfunc[self.id](self, self.data_holder, self.demographic_model, values, filename)
