import tensorflow as tf
import numpy as np
import time
from meta_policy_search.utils import logger
import numpy as np

from collections import OrderedDict


class Trainer(object):
    """
    Performs steps of meta-policy search.

     Pseudocode::

            for iter in n_iter:
                sample tasks
                for task in tasks:
                    for adapt_step in num_inner_grad_steps
                        sample trajectories with policy
                        perform update/adaptation step
                    sample trajectories with post-update policy
                perform meta-policy gradient step(s)

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
            envs,
            samplers,
            sample_processor,
            policy,
            n_itr,
            start_itr=0,
            num_inner_grad_steps=1,
            sess=None,
    ):
        self.algo = algo
        self.envs = envs
        self.samplers = samplers
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_inner_grad_steps = num_inner_grad_steps
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        """
        Trains policy on env using algo

        Pseudocode::

            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """
        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables(
            ) if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log(
                    "\n ---------------- Iteration %d ----------------" % itr)
                logger.log(
                    "Sampling set of tasks/goals for this meta-batch...")

                for sampler in self.samplers:
                    sampler.update_tasks()
                self.policy.switch_to_pre_update()  # Switch to pre-update policy

                all_samples_data, all_paths = [], []
                list_sampling_time, list_inner_step_time, list_outer_step_time, list_proc_samples_time = [], [], [], []
                start_total_inner_time = time.time()
                for step in range(self.num_inner_grad_steps+1):
                    logger.log('** Step ' + str(step) + ' **')

                    """ -------------------- Sampling --------------------------"""

                    logger.log("Obtaining samples...")
                    time_env_sampling_start = time.time()

                    sampler = np.random.choice(self.samplers, p=[0.5, 0.5])
                    paths = sampler.obtain_samples(
                        log=True, log_prefix='Step_%d-' % step)
                    list_sampling_time.append(
                        time.time() - time_env_sampling_start)
                    all_paths.append(paths)

                    """ ----------------- Processing Samples ---------------------"""

                    logger.log("Processing samples...")
                    time_proc_samples_start = time.time()
                    samples_data = self.sample_processor.process_samples(
                        paths, log='all', log_prefix='Step_%d-' % step)
                    all_samples_data.append(samples_data)
                    list_proc_samples_time.append(
                        time.time() - time_proc_samples_start)

                    self.log_diagnostics(
                        sum(list(paths.values()), []), prefix='Step_%d-' % step)

                    """ ------------------- Inner Policy Update --------------------"""

                    time_inner_step_start = time.time()
                    if step < self.num_inner_grad_steps:
                        logger.log("Computing inner policy updates...")
                        self.algo._adapt(samples_data)
                    # train_writer = tf.summary.FileWriter('/home/ignasi/Desktop/meta_policy_search_graph',
                    #                                      sess.graph)
                    list_inner_step_time.append(
                        time.time() - time_inner_step_start)
                total_inner_time = time.time() - start_total_inner_time

                time_maml_opt_start = time.time()
                """ ------------------ Outer Policy Update ---------------------"""

                logger.log("Optimizing policy...")
                # This needs to take all samples_data so that it can construct graph for meta-optimization.
                time_outer_step_start = time.time()
                self.algo.optimize_policy(all_samples_data)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', [
                             sampler.total_timesteps_sampled for sampler in self.samplers])

                logger.logkv('Time-OuterStep', time.time() -
                             time_outer_step_start)
                logger.logkv('Time-TotalInner', total_inner_time)
                logger.logkv('Time-InnerStep', np.sum(list_inner_step_time))
                logger.logkv('Time-SampleProc', np.sum(list_proc_samples_time))
                logger.logkv('Time-Sampling', np.sum(list_sampling_time))

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)
                logger.logkv('Time-MAMLSteps', time.time() -
                             time_maml_opt_start)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.envs, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        #self.envs.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)


class KAML_Trainer(object):
    """
    Performs steps of meta-policy search.

     Pseudocode::

            for iter in n_iter:
                sample tasks
                for task in tasks:
                    for adapt_step in num_inner_grad_steps
                        sample trajectories with policy
                        perform update/adaptation step
                    sample trajectories with post-update policy
                perform meta-policy gradient step(s)

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
            algos,
            envs,
            samplers,
            sample_processor,
            policy,
            n_itr,
            start_itr=0,
            num_inner_grad_steps=1,
            sess=None,
            theta_count=2,
            probs = [0.5, 0.5]
    ):
        print("initialize KAML trainer")
        self.algos = algos
        self.theta_count = theta_count

        self.envs = envs
        self.samplers = samplers
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.probs = probs
        
        assert len(samplers) == len(probs), "len(samplers) = {} != {} = len(probs)".format(len(samplers), len(probs))
        
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        """
        Trains policy on env using algo

        Pseudocode::

            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """
        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables(
            ) if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log(
                    "\n ---------------- Iteration %d ----------------" % itr)
                logger.log(
                    "Sampling set of tasks/goals for this meta-batch...")

                for sampler in self.samplers:
                    sampler.update_tasks()
                self.policy.switch_to_pre_update()  # Switch to pre-update policy

                all_samples_data, all_paths, algo_all_samples = [], [], []
                list_sampling_time, list_inner_step_time, list_outer_step_time, list_proc_samples_time = [], [], [], []
                start_total_inner_time = time.time()
                inner_loop_losses = []
                for step in range(self.num_inner_grad_steps+1):
                    logger.log('** Step ' + str(step) + ' **')

                    """ -------------------- Sampling --------------------------"""

                    logger.log("Obtaining samples...")
                    time_env_sampling_start = time.time()
                    
#                     sampler = np.random.choice(self.samplers, p=[0.5, 0.5])
#                     paths = sampler.obtain_samples(
#                         log=True, log_prefix='Step_%d-' % step)
                    
                    initial_paths = [sampler.obtain_samples(
                        log=True, log_prefix='Step_%d-' % step) for sampler in self.samplers]

#                     print(len(initial_paths[0]), self.probs, list(range(len(initial_paths))))
                    paths = OrderedDict()
                    for i in range(len(initial_paths[0])):#, Paths in enumerate(zip(*initial_paths)):
                        paths[i] = initial_paths[np.random.choice(list(range(len(initial_paths))), p = self.probs)][i]
                        
#                     assert 1 == 2, str(paths)
                    
                    list_sampling_time.append(time.time() - time_env_sampling_start)
                    all_paths.append(paths)

                    """ ----------------- Processing Samples ---------------------"""

                    logger.log("Processing samples...")
                    time_proc_samples_start = time.time()
                    samples_data = self.sample_processor.process_samples(
                        paths, log='all', log_prefix='Step_%d-' % step)
                    all_samples_data.append(samples_data)
                    list_proc_samples_time.append(
                        time.time() - time_proc_samples_start)

                    self.log_diagnostics(
                        sum(list(paths.values()), []), prefix='Step_%d-' % step)

                    """ ------------------- Inner Policy Update --------------------"""
                    if step < self.num_inner_grad_steps:
                        inner_loop_losses = []

                    for algo in self.algos[:self.theta_count]:
                        time_inner_step_start = time.time()
                        if step < self.num_inner_grad_steps:
                            logger.log("Computing inner policy updates...")
                            logger.log("len(samples_data) = {}".format(
                                len(samples_data)))
                            loss_list = algo._adapt(samples_data)
                            inner_loop_losses.append(loss_list)

                    algo_batches = [[] for _ in range(self.theta_count)]

                    indices = np.argmin(inner_loop_losses, axis=0)
                    for i in range(len(samples_data)):
                        index = indices[i]
                        algo_batches[index].append((i, samples_data[i]))

                    algo_all_samples.append(algo_batches)

                    list_inner_step_time.append(
                        time.time() - time_inner_step_start)
                total_inner_time = time.time() - start_total_inner_time

                time_maml_opt_start = time.time()
                """ ------------------ Outer Policy Update ---------------------"""

                logger.log("Optimizing policy...")
                # This needs to take all samples_data so that it can construct graph for meta-optimization.
                time_outer_step_start = time.time()
                for index in range(self.theta_count):
                    all_samples_index_data = [algo_batches[index]
                                              for algo_batches in algo_all_samples]
                    self.algos[index].optimize_policy(all_samples_index_data)

                """ ------------------- Logging Stuff --------------------------"""
                logger.logkv('Itr', itr)
                logger.logkv('n_timesteps', [
                             sampler.total_timesteps_sampled for sampler in self.samplers])

                logger.logkv('Time-OuterStep', time.time() -
                             time_outer_step_start)
                logger.logkv('Time-TotalInner', total_inner_time)
                logger.logkv('Time-InnerStep', np.sum(list_inner_step_time))
                logger.logkv('Time-SampleProc', np.sum(list_proc_samples_time))
                logger.logkv('Time-Sampling', np.sum(list_sampling_time))

                logger.logkv('Time', time.time() - start_time)
                logger.logkv('ItrTime', time.time() - itr_start_time)
                logger.logkv('Time-MAMLSteps', time.time() -
                             time_maml_opt_start)

                logger.log("Saving snapshot...")
                params = self.get_itr_snapshot(itr)
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.envs, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        #self.envs.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
