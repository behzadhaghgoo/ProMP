import os

import tensorflow as tf
import numpy as np
import time
from maml_zoo.logger import logger
from maml_zoo.utils import utils


class Trainer(object):
    """
    Performs steps for MAML
    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) : 
        sample_processor (SampleProcessor) : 
        baseline (Baseline) : 
        policy (Policy) : 
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
    """
    def __init__(
            self,
            algo,
            env,
            sampler,
            sample_processor,
            policy,
            n_itr,
            start_itr=0,
            task=None,
            sess=None,
            meta_batch_size=0,
            pkl=None,
            name='',
            ):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.task = task
        if sess is None:
            sess = tf.Session()
        self.sess = sess
        self.pkl = pkl

        # open a csv file to log all
        # evaluation
        test_csv = '/root/code/ProMP/test_rl2_{}.csv'.format(name)
        if os.path.exists(test_csv):
            self.test_csv = open(test_csv, 'a+')
        else:
            self.test_csv = open(test_csv, 'a+')
            head = 'Itr,'
            for meta_batch in range(meta_batch_size):
                meta_head = 'Task{},{},{},{},'.format(meta_batch, 'AverageDiscountedReturn', 'UndiscountedReturn', 'SuccessRate')
                head += meta_head
            head = head[:-1] + '\n'
            print(head)
            self.test_csv.write(head)
            self.test_csv.flush()

    def train(self, test_time=False):
        """
        Trains policy on env using algo
        Pseudocode:
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """
        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                self.task = self.env.sample_tasks(self.sampler.meta_batch_size)
                self.sampler.set_tasks(self.task)
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Sampling set of tasks/goals for this meta-batch...")

                """ -------------------- Sampling --------------------------"""

                logger.log("Obtaining samples...")
                time_env_sampling_start = time.time()
                paths = self.sampler.obtain_samples(log=True, log_prefix='train-')
                sampling_time = time.time() - time_env_sampling_start

                """ ----------------- Processing Samples ---------------------"""

                logger.log("Processing samples...")
                time_proc_samples_start = time.time()
                samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='train-')
                proc_samples_time = time.time() - time_proc_samples_start

                if test_time:
                    logger.log("Done with collecting and processing test data.")
                    logger.log("Saving per-task data now.")

                    rollout_task_paths = []
                    for i in range(len(paths.keys())):
                        d = dict(returns=[], success=[], rewards=[])
                        rollout_task_paths.append(d)
                    for meta_task, meta_path in paths.items():
                        for rollout_idx, rollout_path in enumerate(meta_path):
                            rollout_path["returns"] = utils.discount_cumsum(rollout_path["rewards"], self.sample_processor.discount)
                            rollout_task_paths[meta_task]["returns"].extend(rollout_path["returns"])
                            rollout_task_paths[meta_task]["rewards"].append(rollout_path["rewards"])
                            if "success" in rollout_path["env_infos"]:
                                if np.sum(rollout_path["env_infos"]["success"]) >= 1:
                                    rollout_task_paths[meta_task]["success"].append(1)
                                else:
                                    rollout_task_paths[meta_task]["success"].append(0)
                            if "task_name" in rollout_path['env_infos']:
                                rollout_task_paths[meta_task]['task_name'] = rollout_path['env_infos']['task_name'][0]

                    line = '{},'.format(str(itr) + '_{}'.format(self.pkl))
                    for meta_task, meta_task_rollout in enumerate(rollout_task_paths):
                        if not meta_task_rollout["success"]:
                            success_rate = None
                        else:
                            success_rate = np.mean(meta_task_rollout["success"])
                        average_discounted_return = np.mean(meta_task_rollout["returns"])
                        undiscounted_returns = np.mean(np.sum(meta_task_rollout["rewards"], axis=1))
                        line = line + '{},{},{},{},'.format(meta_task_rollout["task_name"], average_discounted_return, undiscounted_returns, success_rate)
                    line = line[:-1] + '\n'
                    print(line)
                    self.test_csv.write(line)
                    self.test_csv.flush()
                    logger.dumpkvs()
                    return

                self.log_diagnostics(sum(paths.values(), []), prefix='train-')

                """ ------------------ Policy Update ---------------------"""

                logger.log("Optimizing policy...")
                # This needs to take all samples_data so that it can construct graph for meta-optimization.
                time_optimization_step_start = time.time()
                self.algo.optimize_policy(samples_data)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', self.sampler.total_timesteps_sampled)

                logger.logkv('Time-Optimization', time.time() - time_optimization_step_start)
                logger.logkv('Time-SampleProc', np.sum(proc_samples_time))
                logger.logkv('Time-Sampling', sampling_time)

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()
                if itr == 0:
                    sess.graph.finalize()

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
