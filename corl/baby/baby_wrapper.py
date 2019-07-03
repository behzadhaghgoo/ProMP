import gym
import numpy as np
from maml_zoo.utils import Serializable


class BabyModeWrapper(gym.Wrapper, Serializable):

    def __init__(self, env):
        Serializable.quick_init(self, locals())
        super().__init__(env)
        self.num_tasks = len(env.tasks)

    '''
    MAML sampler api
    
    In baby mode, tasks will be reset in the environment reset
    function so we ignore task sampling/setting

    Tasks should be sampled before construnction.
    '''
    def sample_tasks(self, meta_batch_size):
        return np.random.randint(0, self.num_tasks, size=meta_batch_size)
    
    def set_task(self, task):
        self.env.task_idx = task

    def log_diagnostics(self, paths, prefix):
        pass
