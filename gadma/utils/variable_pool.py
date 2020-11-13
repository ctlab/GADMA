from .variables import Variable
import copy


class VariablePool(list):
    """
    Class for list of variables. All variables should have different names.

    :param lst: list of the variables.
    """
    def __init__(self, lst=None):
        self.names = set()
        if lst is not None:
            for item in lst:
                self.append(item)

    def fix_pickling(self):
        if not hasattr(self, 'names'):
            self.names = set()
            for item in self:
                self.names.add(item.name)

    def check_type(self, item):
        """
        Raises ValueError if ``item`` is not Variable.
        """
        if not isinstance(item, Variable):
            raise ValueError(f"Items of VariablePool could be Variables only.")

    def append(self, item):
        """
        Append new ``item`` to pool.
        """
        self.fix_pickling()
        self.check_type(item)
        if item.name not in self.names:
            self.names.add(item.name)
            super(VariablePool, self).append(item)
        else:
            raise NameError(f"VariablePool has already a Variable with "
                            f"the same name ({item.name}).")

    def extend(self, items):
        """
        Extend pool with other list.
        """
        for item in items:
            self.append(item)

    def __setitem__(self, key, item):
        self.fix_pickling()
        if isinstance(key, slice):
            remove_names = []
            for value in self[key]:
                remove_names.append(value.name)
            for value in item:
                self.check_type(value)
                if value.name in self.names and value.name not in remove_names:
                    raise NameError(f"VariablePool has already a Variable"
                                    f" with the same name ({key}).")
            for name in remove_names:
                self.names.remove(name)
            for value in item:
                self.names.add(value.name)
            return super(VariablePool, self).__setitem__(key, item)

        self.check_type(item)
        if item.name not in self.names or item.name == self[key].name:
            self.names.remove(self[key].name)
            self.names.add(item.name)
            return super(VariablePool, self).__setitem__(key, item)
        else:
            raise NameError(f"VariablePool has already a Variable with "
                            f"the same name ({key}).")

    def __delitem__(self, key):
        self.fix_pickling()
        if isinstance(key, slice):
            for value in self[key]:
                self.names.remove(value.name)
        else:
            value = self[key]
            self.names.remove(value.name)
        super(VariablePool, self).__delitem__(key)

    def __copy__(self):
        newone = type(self)(self)
        return newone

    def __deepcopy__(self, memo):
        lst = list()
        for var in self:
            lst.append(copy.deepcopy(var, memo))
        newone = type(self)(lst)
        return newone
