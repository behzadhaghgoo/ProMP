from half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from sawyer_push import SawyerPushEnv
from meta_policy_search.envs.normalized_env import normalize
# TODO: should normalization be done here?

class MixedEnv(MetaEnv): #TODO parents
    def __init__(self, mujoco_ratio = 0.5, envs = ["HalfCheetahRandDirecEnv", "SawyerPushEnv"]):
        self.mujoco_ratio = mujoco_ratio
        # instantiate envs
        self.envs = [globals()[env]() for env in envs] 
        # apply normalize wrapper to envs
        self.envs = [normalize(env) for env in self.envs]
    
    def sample_tasks(self, n_tasks):
        
        return self.sawyer_sample_goals(n_tasks)

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        
        

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.sawyer_get_goal()

