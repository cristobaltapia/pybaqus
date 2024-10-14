"""
Define a step object
"""
import numpy as np

class Step(object):

    """Class to contain step information and methods

    Parameters
    ----------
    model : TODO
    n : int
    data : dict

    """

    def __init__(self, model, n, data):
        """TODO: to be defined.


        """
        self._model = model
        self._n = n
        self._data = data
        self.tot_time = data["total time"]
        self.max_creep = data["max creep"]
        self.sol_apl = data["solution amplitude"]
        self.proc_type = data["procedure type"]
        self.step_n = data["step number"]
        self.lin_pert = data["linear perturbation"]
        self.freq = data["frequency"]
        self.subheading = data["subheading"]

        self.step_time: list = [data["step time"]]
        self.load_prop: list = [data["load proportionality"]]
        self.time_inc: list = [data["time increment"]]
        self.increments: list = [data["increment number"]]

    def add_increment(self, inc, time_inc, step_time, load_prop):
        """Add increment to the step object

        Parameters
        ----------
        inc : TODO

        """
        self.step_time.append(step_time)
        self.load_prop.append(load_prop)
        self.increments.append(inc)
        self.time_inc.append(time_inc)

        return 1
