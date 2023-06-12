from typing import Union, Tuple, List, Sequence, Optional, Callable
import uuid

from eolearn.core import EOTask, EONode, FeatureType, EOPatch


class ConditionalTask(EOTask):
    def __init__(self,
                 conditional: Callable,
                 eotask: EOTask,
                 indicate_condition: Optional[str] = None,
                 remove_indicator: Optional[str] = None,
                 verbose: bool = False):
        """
        Wrapper class around an arbitrary EOTask. Performs the original execute method if conditional returns True.
        Passes the input EOPatch unchanged if conditional returns false.
        :param conditional: function taking in as argument an EOPatch; returns True or False
        :param eotask: Instance of EOTask to be wrapped
        :param indicate_condition: Flag indicating the result of conditional. Stored in EOPatch meta info.
        Predominantly used in conditional chaining.
        :param remove_indicator: Feature name of indicator to remove. Predominantly used in conditional chaining.
        :param verbose: Whether to print result of conditional.
        """
        self.conditional = conditional
        self.task = eotask

        self._indicate_condition = indicate_condition
        self._remove_indicator = remove_indicator

        self.verbose = verbose

    def execute(self, eopatch: EOPatch, **kwargs) -> EOPatch:
        if self.conditional(eopatch):
            if self.verbose:
                print(f"{self.__class__.__name__} ({self.task.__class__.__name__}) :"
                      f"Conditional positive, executing nested task.")
            if self._indicate_condition: eopatch[(FeatureType.META_INFO, self._indicate_condition)] = 1
            if self._remove_indicator is not None: del eopatch[(FeatureType.META_INFO, self._remove_indicator)]
            return self.task.execute(eopatch, **kwargs)
        else:
            if self.verbose:
                print(f"{self.__class__.__name__} ({self.task.__class__.__name__}): "
                      f"Conditional negative, returning input EOPatch.")
            if self._indicate_condition: eopatch[(FeatureType.META_INFO, self._indicate_condition)] = 0
            if self._remove_indicator is not None: del eopatch[(FeatureType.META_INFO, self._remove_indicator)]
            return eopatch


def conditional_node_chain(conditional: Callable, *tasks: Union[EOTask, Tuple[EOTask, str]],
                           inputs: Optional[Sequence[EONode]] = (), name="cond", **kwargs) -> List[EONode]:
    """
    Utility function chaining a series of EOTasks and conditioning their execution on one initial conditional.
    If the initial conditional is negative, the input EOPatch will be passed on unchanged, if positive, the whole
    chain of nodes is executed.
    :param conditional: The initial conditional deciding whether to execute
    :param tasks: A sequence of EOTasks to be wrapped in ConditionalTasks and EONodes and chained linearly
    :param inputs: Inputs passed to the initial EONode
    :param name: Prefix to pass to the name parameter of the resulting EONodes
    :param kwargs: Further arguments to be passed to the ConditionalTasks
    :return: A list of EONodes with ConditionalTasks as their tasks, all conditioned on the initial conditional
    """
    conditional_nodes = []
    condition_indicator = str(uuid.uuid4())

    for i, task_ in enumerate(tasks):
        if isinstance(task_, EOTask):
            task = task_
            name_task = ""
        else:
            task, name_task = task_
            name_task = "\n" + name_task

        if len(tasks) == 1:
            cond_task = ConditionalTask(conditional, task, **kwargs)
        elif i == 0:
            cond_task = ConditionalTask(conditional, task, indicate_condition=condition_indicator, **kwargs)
        else:
            conditional = lambda patch: patch[(FeatureType.META_INFO, condition_indicator)] == 1
            if i == (len(tasks) - 1):
                cond_task = ConditionalTask(conditional, task, remove_indicator=condition_indicator, **kwargs)
            else:
                cond_task = ConditionalTask(conditional, task, **kwargs)

        cond_node = EONode(cond_task, inputs=inputs, name=name + f" ({i}){name_task}")
        conditional_nodes.append(cond_node)
        inputs = [cond_node]
    return conditional_nodes
