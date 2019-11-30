from multiworld.envs.mujoco.sawyer_xyz.push.sawyer_push import SawyerPushEnv as SawyerEnv
from multiworld.core.flat_goal_env import FlatGoalEnv
import numpy as np
from meta_policy_search.envs.base import MetaEnv
from meta_policy_search.utils import logger


class SawyerPushEnv(FlatGoalEnv, MetaEnv):
    """
    Wrapper for SawyerPushEnv from multiworld envs, using our method headers
    """
    def __init__(self, *args, **kwargs):
        self.sawyer_quick_init(locals())
        sawyer_env = SawyerEnv(*args, **kwargs)
        FlatGoalEnv.__init__(self, sawyer_env, obs_keys=['state_observation'], goal_keys=['state_desired_goal'])

    def sample_tasks(self, n_tasks):
        return self.sawyer_sample_goals(n_tasks)

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        return self.sawyer_set_goal(task)

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.sawyer_get_goal()

