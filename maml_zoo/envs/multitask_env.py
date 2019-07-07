from cached_property import cached_property
from maml_zoo.utils import Serializable
import gym
import numpy as np

from multiworld.envs.mujoco.random_init_wrapper import generate_random_init_configs, RandomInitWrapper


class MultiTaskEnv(gym.Env, Serializable):
    def __init__(self,
                 task_env_cls=None,
                 task_args=None,
                 task_kwargs=None,):
        Serializable.quick_init(self, locals())
        self._task_envs = [
            task_env_cls(*t_args, **t_kwargs)
            for t_args, t_kwargs in zip(task_args, task_kwargs)
        ]
        self._active_task = None

    def reset(self, **kwargs):
        return self.active_env.reset(**kwargs)

    @property
    def action_space(self):
        return self.active_env.action_space

    @property
    def observation_space(self):
        return self.active_env.observation_space

    def step(self, action):
        obs, reward, done, info = self.active_env.step(action)
        info['task'] = self.active_task_one_hot
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self.active_env.render(*args, **kwargs)

    def close(self):
        for env in self._task_envs:
            env.close()

    @property
    def task_space(self):
        n = len(self._task_envs)
        one_hot_ub = np.ones(n)
        one_hot_lb = np.zeros(n)
        return gym.spaces.Box(one_hot_lb, one_hot_ub, dtype=np.float32)

    @property
    def active_task(self):
        return self._active_task

    @property
    def active_task_one_hot(self):
        one_hot = np.zeros(self.task_space.shape)
        t = self.active_task or 0
        one_hot[t] = self.task_space.high[t]
        return one_hot

    @property
    def active_env(self):
        return self._task_envs[self.active_task or 0]

    @property
    def num_tasks(self):
        return len(self._task_envs)

    '''
    API's for MAML Sampler
    '''
    def sample_tasks(self, meta_batch_size):
        return np.random.randint(0, self.num_tasks, size=meta_batch_size)
    
    def set_task(self, task):
        self._active_task = task

    def log_diagnostics(self, paths, prefix):
        pass


class MultiClassMultiTaskEnv(MultiTaskEnv):
    def __init__(self,
                 task_env_cls_dict=None,
                 task_args_kwargs=None,
                 sampled_tasks=None,
                 sample_all=True,
                 random_init=False,
                 n_random_init=5):
        Serializable.quick_init(self, locals())

        assert len(task_env_cls_dict.keys()) == len(task_args_kwargs.keys())
        for k in task_env_cls_dict.keys():
            assert k in task_args_kwargs

        self._task_envs = []
        self._task_names = []
        self._sampled_all = sample_all

        for task, env_cls in task_env_cls_dict.items():
            task_args = task_args_kwargs[task]['args']
            task_kwargs = task_args_kwargs[task]['kwargs']
            task_env = env_cls(*task_args, **task_kwargs)
            self._task_envs.append(task_env)
            self._task_names.append(task)

        self._active_task = 0


    def set_task(self, task):
        self._active_task = task % len(self._task_envs)

    def sample_tasks(self, meta_batch_size):
        if self._sampled_all:
            return [i for i in range(meta_batch_size)]
        else:
            return np.random.randint(0, self.num_tasks, size=meta_batch_size)

    def step(self, action):
        obs, reward, done, info = self.active_env.step(action)
        if 'task_type' in dir(self.active_env):
            name = '{}-{}'.format(str(self.active_env.__class__.__name__), self.active_env.task_type)
        else:
            name = str(self.active_env.__class__.__name__)
        info['task_name'] = name
        return obs, reward, done, info
