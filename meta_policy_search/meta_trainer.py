import tensorflow as tf
import numpy as np
import time
from meta_policy_search.utils import logger
import numpy as np

from collections import OrderedDict


class Timer():
    """Timer Class."""

    def __init__(self):
        """Initialize timer and lap counter."""
        self.time = time.time()
        self.counter = 0

    def time_elapsed(self):
        """
        Return time elapsed since last interaction with the timer
        (Either starting it or getting time elapsed).
        """
        self.counter += 1
        delta_t = time.time() - self.time
        print(self.counter, delta_t)
        self.time = time.time()
        return delta_t

    def start(self):
        """Restart timer"""
        self.time = time.time()
        self.counter = 0


# class Trainer(object):
#     """
#     Performs steps of meta-policy search.

#      Pseudocode::

#             for iter in n_iter:
#                 sample tasks
#                 for task in tasks:
#                     for adapt_step in num_inner_grad_steps
#                         sample trajectories with policy
#                         perform update/adaptation step
#                     sample trajectories with post-update policy
#                 perform meta-policy gradient step(s)

#     Args:
#         algo (Algo) :
#         env (Env) :
#         sampler (Sampler) :
#         sample_processor (SampleProcessor) :
#         baseline (Baseline) :
#         policy (Policy) :
#         n_itr (int) : Number of iterations to train for
#         start_itr (int) : Number of iterations policy has already trained for, if reloading
#         num_inner_grad_steps (int) : Number of inner steps per maml iteration
#         sess (tf.Session) : current tf session (if we loaded policy, for example)
#     """

#     def __init__(
#             self,
#             algo,
#             envs,
#             env_ids,
#             samplers,
#             sample_processor,
#             policy,
#             n_itr,
#             start_itr=0,
#             num_inner_grad_steps=1,
#             sess=None,
#     ):
#         self.algo = algo
#         self.envs = envs
#         self.env_ids = env_ids
#         self.samplers = samplers
#         self.sample_processor = sample_processor
#         self.baseline = sample_processor.baseline
#         self.policy = policy
#         self.n_itr = n_itr
#         self.start_itr = start_itr
#         self.num_inner_grad_steps = num_inner_grad_steps
#         self.saver = tf.train.Saver()
#         if sess is None:
#             sess = tf.Session()
#         self.sess = sess

#     def train(self):
#         """
#         Trains policy on env using algo

#         Pseudocode::

#             for itr in n_itr:
#                 for step in num_inner_grad_steps:
#                     sampler.sample()
#                     algo.compute_updated_dists()
#                 algo.optimize_policy()
#                 sampler.update_goals()
#         """
#         with self.sess.as_default() as sess:

#             # initialize uninitialized vars  (only initialize vars that were not loaded)
#             uninit_vars = [var for var in tf.global_variables(
#             ) if not sess.run(tf.is_variable_initialized(var))]
#             sess.run(tf.variables_initializer(uninit_vars))

#             start_time = time.time()
#             for itr in range(self.start_itr, self.n_itr):
#                 itr_start_time = time.time()
#                 logger.log(
#                     "\n ---------------- Iteration %d ----------------" % itr)
#                 logger.log(
#                     "Sampling set of tasks/goals for this meta-batch...")

#                 for sampler in self.samplers:
#                     sampler.update_tasks()
#                 self.policy.switch_to_pre_update()  # Switch to pre-update policy

#                 all_samples_data, all_paths = [], []
#                 list_sampling_time, list_inner_step_time, list_outer_step_time, list_proc_samples_time = [], [], [], []
#                 start_total_inner_time = time.time()
#                 for step in range(self.num_inner_grad_steps+1):
#                     logger.log('** Step ' + str(step) + ' **')

#                     """ -------------------- Sampling --------------------------"""

#                     logger.log("Obtaining samples...")
#                     time_env_sampling_start = time.time()

#                     sampler = np.random.choice(self.samplers, p=[0.5, 0.5])
#                     paths = sampler.obtain_samples(
#                         log=True, log_prefix='Step_%d-' % step)
#                     list_sampling_time.append(
#                         time.time() - time_env_sampling_start)
#                     all_paths.append(paths)

#                     """ ----------------- Processing Samples ---------------------"""

#                     logger.log("Processing samples...")
#                     time_proc_samples_start = time.time()
#                     samples_data = self.sample_processor.process_samples(
#                         paths, log='all', log_prefix='Step_%d-' % step)
#                     all_samples_data.append(samples_data)
#                     list_proc_samples_time.append(
#                         time.time() - time_proc_samples_start)

#                     self.log_diagnostics(
#                         sum(list(paths.values()), []), prefix='Step_%d-' % step)

#                     """ ------------------- Inner Policy Update --------------------"""

#                     time_inner_step_start = time.time()
#                     if step < self.num_inner_grad_steps:
#                         logger.log("Computing inner policy updates...")
#                         self.algo._adapt(samples_data)
#                     # train_writer = tf.summary.FileWriter('/home/ignasi/Desktop/meta_policy_search_graph',
#                     #                                      sess.graph)
#                     list_inner_step_time.append(
#                         time.time() - time_inner_step_start)
#                 total_inner_time = time.time() - start_total_inner_time

#                 time_maml_opt_start = time.time()
#                 """ ------------------ Outer Policy Update ---------------------"""

#                 logger.log("Optimizing policy...")
#                 # This needs to take all samples_data so that it can construct graph for meta-optimization.
#                 time_outer_step_start = time.time()
#                 self.algo.optimize_policy(all_samples_data)

#                 """ ------------------- Logging Stuff --------------------------"""
#                 logger.logkv('Itr', itr)
#                 logger.logkv('n_timesteps', [
#                              sampler.total_timesteps_sampled for sampler in self.samplers])

#                 logger.logkv('Time-OuterStep', time.time() -
#                              time_outer_step_start)
#                 logger.logkv('Time-TotalInner', total_inner_time)
#                 logger.logkv('Time-InnerStep', np.sum(list_inner_step_time))
#                 logger.logkv('Time-SampleProc', np.sum(list_proc_samples_time))
#                 logger.logkv('Time-Sampling', np.sum(list_sampling_time))

#                 logger.logkv('Time', time.time() - start_time)
#                 logger.logkv('ItrTime', time.time() - itr_start_time)
#                 logger.logkv('Time-MAMLSteps', time.time() -
#                              time_maml_opt_start)

#                 logger.log("Saving snapshot...")
#                 params = self.get_itr_snapshot(itr)
#                 logger.save_itr_params(itr, params)
#                 logger.log("Saved")

#                 logger.dumpkvs()

#         logger.log("Training finished")
#         self.saver.save(sess, '{}'.format(self.env_ids))
#         self.sess.close()

#     def get_itr_snapshot(self, itr):
#         """
#         Gets the current policy and env for storage
#         """
#         return dict(itr=itr, policy=self.policy, env=self.envs, baseline=self.baseline)

#     def log_diagnostics(self, paths, prefix):
#         # TODO: we aren't using it so far
#         #self.envs.log_diagnostics(paths, prefix)
#         self.policy.log_diagnostics(paths, prefix)
#         self.baseline.log_diagnostics(paths, prefix)


class KAML_Test_Trainer(object):
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
            policies,
            n_itr,
            probs,
            start_itr=0,
            num_inner_grad_steps=1,
            sess=None,
            theta_count=2,
            multi_maml = False,
            phi_test = False,
            switch_thresh = False,
            mode_name = None,
    ):
        print("initialize KAML test trainer")
        self.algos = algos
        self.theta_count = theta_count

        self.envs = envs
        self.samplers = samplers
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policies = policies
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.probs = probs
        self.saver = tf.train.Saver()
        self.num_envs = len(envs)
        self.meta_batch_size = self.samplers[0].meta_batch_size

        self.multi_maml = multi_maml
        self.phi_test = phi_test
        self.switch_thresh = switch_thresh
        self.mode_name = mode_name
        assert len(samplers) == len(
            probs), "len(samplers) = {} != {} = len(probs)".format(len(samplers), len(probs))

        if sess is None:
            sess = tf.Session()
        self.sess = sess

        self.timer = Timer()

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

        self.timer.start()

        # Set experiment parameters 
        multi_maml = self.multi_maml
        phi_test = self.phi_test
        switch_thresh = self.switch_thresh
        
        #######################################
        #######################################
        #######################################
        
        load_checkpoint = False
        print("load_checkpoint is {}".format(load_checkpoint))
        
        #######################################
        ############# Loader Code #############
        #######################################
        
        checkpoint_name = None
        if load_checkpoint: 
            assert checkpoint_name, "Provide checkpoint name."
        print("loading from checkpoint {}".format(checkpoint_name))
        
        #######################################
        #######################################
        #######################################

        with self.sess.as_default() as sess:

            if load_checkpoint:
                saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_name))
                saver.restore(sess,checkpoint_name)
                
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables(
            ) if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
            
#                 logger.logkv("Iteration time elapsed", self.timer.time_elapsed())
                
                itr_start_time = time.time()
                logger.log(
                    "\n ---------------- Iteration %d ----------------" % itr)
                logger.log("Mode: {}".format(self.mode_name))
                logger.log(
                    "Sampling set of tasks/goals for this meta-batch...")

                # Update tasks if not in the initial phi phase.
                if (not phi_test) or itr > switch_thresh:
                    for sampler in self.samplers:
                        sampler.update_tasks()

                # all_inner_loop_losses = []
                # all_samples_data = []

                # shape : num_algos, num_inner_steps, num_tasks, ..
                all_algo_all_samples_data = [[]
                                             for _ in range(self.theta_count)]
                # shape : num_algos, num_tasks
                all_algo_inner_loop_losses = []

                true_indices = []
                for task_ind in range(self.meta_batch_size):
                    index = np.random.choice(
                        list(range(self.num_envs)), p=self.probs)
                    true_indices.append(index)

                # For each theta in thetas, we obtain trajectories from the same tasks from both environments
                algo_samples_reward_data = []
                for a_ind, algo in enumerate(self.algos[:self.theta_count]):
                    # algo_inner_loop_losses = []
                    # algo_all_samples_data = []

                    policy = algo.policy
                    policy.switch_to_pre_update()  # Switch to pre-update policy

                    # all_samples_data, all_paths, algo_all_samples = [], [], []
                    list_sampling_time, list_inner_step_time, list_outer_step_time, list_proc_samples_time = [], [], [], []
                    start_total_inner_time = time.time()
                    algo_inner_loop_losses = []

                    for step in range(self.num_inner_grad_steps+1):
                        logger.log('** Step ' + str(step) + ' **')

                        """ -------------------- Sampling --------------------------"""

                        logger.log("Obtaining samples...")
                        time_env_sampling_start = time.time()

                        # Meta-sampler's obtain_samples function now takes as input policy since we need trajectories for each policy
                        initial_paths = [sampler.obtain_samples(
                            policy=policy, log=True, log_prefix='Step_%d-' % step) for sampler in self.samplers]

                        # get paths no matter step == 0 or 1
                        paths = OrderedDict()
                        for task_ind in range(self.meta_batch_size):
                            index = true_indices[task_ind]
                            paths[task_ind] = initial_paths[index][task_ind]

                        if step == self.num_inner_grad_steps:
                            algo_samples_reward_data.append(paths)

                        list_sampling_time.append(
                            time.time() - time_env_sampling_start)

                        """ ----------------- Processing Samples ---------------------"""

                        logger.log("Processing samples...")
                        time_proc_samples_start = time.time()
                        samples_data = self.sample_processor.process_samples(
                            paths, log='all', log_prefix='Step_%d-' % step)                            
                        # (number of inner updates, meta_batch_size)

                        all_algo_all_samples_data[a_ind].append(samples_data)

                        list_proc_samples_time.append(
                            time.time() - time_proc_samples_start)

                        self.log_diagnostics(
                            sum(list(paths.values()), []), prefix='Step_%d-' % step)

                        """ ------------------- Inner Policy Update --------------------"""
                        if step == self.num_inner_grad_steps:
                            # In the last inner_grad_step, append inner loop losses of this algo to inner_loop_losses
                            all_algo_inner_loop_losses.append(
                                algo_inner_loop_losses)

                        time_inner_step_start = time.time()
                        if step < self.num_inner_grad_steps:
                            logger.log("Computing inner policy updates...")
                            algo_inner_loop_losses, _ = algo._adapt(
                                samples_data)

                        list_inner_step_time.append(
                            time.time() - time_inner_step_start)
                    total_inner_time = time.time() - start_total_inner_time

                time_maml_opt_start = time.time()
                """ ------------------ Outer Policy Update ---------------------"""

                logger.log("Optimizing policy...")

                time_outer_step_start = time.time()

                # true_indices: len(num_tasks)
                # all_algo_all_samples_data: num_algos, num_inner_steps, num_tasks, ..
                # all_algo_inner_loop_losses : num_algos, num_tasks

                # algo.optimize_policy(data) -> data shape: num_inner_steps, k (< num_tasks), ...

                # inner_loop_losses[i][j] = loss for task j for algo i
                all_algo_all_samples_data = np.array(all_algo_all_samples_data)
                all_algo_inner_loop_losses = np.array(
                    all_algo_inner_loop_losses)
                true_indices = np.array(true_indices)

                if (multi_maml or phi_test) and itr < switch_thresh:
                    which_algo = true_indices
                else:
                    which_algo = np.argmin(
                        all_algo_inner_loop_losses, axis=0)  # length num_tasks
                # print("which_algo shape: ", which_algo.shape)

                # For each algo, do outer update
                relevant_paths = []
                for a_ind, algo in enumerate(self.algos[:self.theta_count]):
                    # Get all indices of data from tasks that were assigned to this algo
                    relevant_data_indices = (which_algo == a_ind)
                    # print("relevant_data_indices", relevant_data_indices.shape)
                    relevant_data_indices = np.nonzero(
                        relevant_data_indices)[0]
                    # print("relevant_data_indices", relevant_data_indices.shape)
                    print(
                        "all_algo_all_samples_data[a_ind, :, relevant_data_indices]")

                    # Fill the batch to make the shape right.
                    x = (all_algo_all_samples_data[a_ind, :, list(
                        relevant_data_indices)])  # 21 x 2
                    
#                     print(algo_samples_reward_data)
                    path_list = algo_samples_reward_data[a_ind]
                    for index in relevant_data_indices:
                        relevant_paths.append(path_list[index])

                    # if in the initial phase of phi_test, cut x to one example.
                    if phi_test and itr < switch_thresh:
                        print("initial x.shape = {}".format(x.shape))
                        x = x[0:1,:]
                        print("converted to x.shape = {}".format(x.shape))

                    difference = self.meta_batch_size - x.shape[0]
                    sample_indices = np.random.choice(
                        x.shape[0], difference, replace=True)

                    new_x = np.concatenate([x, x[sample_indices]], axis=0)
                    np.random.shuffle(new_x)

                    # print("optimize policy input", new_x.shape)
                    algo.optimize_policy(new_x.T)

                self.sample_processor._log_path_stats(relevant_paths, log='reward', log_prefix='Step_2-')

                clustering_score = np.abs(
                    np.mean(np.abs(true_indices - which_algo)) - 0.5) * 2.0
                logger.logkv('Clustering Score', clustering_score)

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
                if itr % 25 == 0:
                    print("Saving model...")
#                     self.saver.save(sess, './MultiMaml_{}_PhiTest_{}_Iteration_{}'.format(multi_maml, phi_test, itr))
                    self.saver.save(sess, './{}_Iteration_{}'.format("_".join(self.mode_name.split()), itr))
                logger.save_itr_params(itr, params)
                logger.log("Saved")

                logger.dumpkvs()

        logger.log("Training finished")
        self.sess.close()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policies=self.policies, env=self.envs, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        #self.envs.log_diagnostics(paths, prefix)
        # self.poli.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)


# class KAML_Trainer(object):
#     """
#     Performs steps of meta-policy search.

#      Pseudocode::

#             for iter in n_iter:
#                 sample tasks
#                 for task in tasks:
#                     for adapt_step in num_inner_grad_steps
#                         sample trajectories with policy
#                         perform update/adaptation step
#                     sample trajectories with post-update policy
#                 perform meta-policy gradient step(s)

#     Args:
#         algo (Algo) :
#         env (Env) :
#         sampler (Sampler) :
#         sample_processor (SampleProcessor) :
#         baseline (Baseline) :
#         policy (Policy) :
#         n_itr (int) : Number of iterations to train for
#         start_itr (int) : Number of iterations policy has already trained for, if reloading
#         num_inner_grad_steps (int) : Number of inner steps per maml iteration
#         sess (tf.Session) : current tf session (if we loaded policy, for example)
#     """

#     def __init__(
#             self,
#             algos,
#             envs,
#             samplers,
#             sample_processor,
#             policies,
#             n_itr,
#             start_itr=0,
#             num_inner_grad_steps=1,
#             sess=None,
#             theta_count=2,
#             probs = [0.5, 0.5]
#     ):
#         print("initialize KAML trainer")
#         self.algos = algos
#         self.theta_count = theta_count

#         self.envs = envs
#         self.samplers = samplers
#         self.sample_processor = sample_processor
#         self.baseline = sample_processor.baseline
#         self.policies = policies
#         self.n_itr = n_itr
#         self.start_itr = start_itr
#         self.num_inner_grad_steps = num_inner_grad_steps
#         self.probs = probs

#         assert len(samplers) == len(
#             probs), "len(samplers) = {} != {} = len(probs)".format(len(samplers), len(probs))

#         if sess is None:
#             sess = tf.Session()
#         self.sess = sess

#         self.timer = Timer()

#     def train(self):
#         """
#         Trains policy on env using algo

#         Pseudocode::

#             for itr in n_itr:
#                 for step in num_inner_grad_steps:
#                     sampler.sample()
#                     algo.compute_updated_dists()
#                 algo.optimize_policy()
#                 sampler.update_goals()
#         """

#         self.timer.start()

#         with self.sess.as_default() as sess:

#             # initialize uninitialized vars  (only initialize vars that were not loaded)
#             uninit_vars = [var for var in tf.global_variables(
#             ) if not sess.run(tf.is_variable_initialized(var))]
#             sess.run(tf.variables_initializer(uninit_vars))

#             # Initial stuff
#             for sampler in self.samplers:
#                 sampler.update_tasks()

#             first_policy = self.algos[0].policy
#             first_policy.switch_to_pre_update()

#             all_samples_data, all_paths, algo_all_samples = [], [], []
#             list_sampling_time, list_inner_step_time, list_outer_step_time, list_proc_samples_time = [], [], [], []
#             start_total_inner_time = time.time()
#             INITIAL_STEPS = 9
#             for step in range(INITIAL_STEPS + 1):
#                 initial_paths = [sampler.obtain_samples(
#                     policy=first_policy, log=True, log_prefix='Step_%d-' % step) for sampler in self.samplers]

#                 true_indices = []
#                 paths = OrderedDict()
#                 # len(self.envs) == len(initial_paths)
#                 # , Paths in enumerate(zip(*initial_paths)):
#                 for i in range(len(initial_paths[0])):
#                     index = np.random.choice(
#                         list(range(len(initial_paths))), p=self.probs)
#                     paths[i] = initial_paths[index][i]

#                     # (number of inner updates, meta_batch_size)
#                     all_paths.append(paths)
#                     """ ----------------- Processing Samples ---------------------"""

#                     logger.log("Processing samples...")
#                     time_proc_samples_start = time.time()
#                     samples_data = self.sample_processor.process_samples(
#                         paths, log='all', log_prefix='Step_%d-' % step)
#                     # (number of inner updates, meta_batch_size)
#                     all_samples_data.append(samples_data)

#                     list_proc_samples_time.append(
#                         time.time() - time_proc_samples_start)

#                     self.log_diagnostics(
#                         sum(list(paths.values()), []), prefix='Step_%d-' % step)

#                     """ ------------------- Inner Policy Update --------------------"""
#                     if step < INITIAL_STEPS:
#                         inner_loop_losses = []
#                         logger.log("Computing inner policy updates...")
#                         phis = self.algos[0]._adapt(samples_data, 1)
#                         # inner_loop_losses.append(loss_list)

#                 """ ------------------ Outer Policy Update ---------------------"""

#                 logger.log("Optimizing policy...")
#                 # This needs to take all samples_data so that it can construct graph for meta-optimization.
#                 time_outer_step_start = time.time()
#                 # all_samples_index_data = [algo_batches[index]
#                 #                          for algo_batches in algo_all_samples]
#                 first_policy.set_params(phis)

#             start_time = time.time()
#             for itr in range(self.start_itr, self.n_itr):

#                 print("\n\n\n\n\n")
#                 self.timer.time_elapsed()
#                 print("\n\n\n\n\n")

#                 itr_start_time = time.time()
#                 logger.log(
#                     "\n ---------------- Iteration %d ----------------" % itr)
#                 logger.log(
#                     "Sampling set of tasks/goals for this meta-batch...")

#                 # Here, we're sampling meta_batch_size / |envs| # of tasks for each environment
#                 for sampler in self.samplers:
#                     sampler.update_tasks()

#                 # For each theta in thetas, we obtain trajectories from the same tasks from both environments
#                 for algo in self.algos[:self.theta_count]:
#                     policy = algo.policy
#                     policy.switch_to_pre_update()  # Switch to pre-update policy

#                     all_samples_data, all_paths, algo_all_samples = [], [], []
#                     list_sampling_time, list_inner_step_time, list_outer_step_time, list_proc_samples_time = [], [], [], []
#                     start_total_inner_time = time.time()
#                     inner_loop_losses = []
#                     for step in range(self.num_inner_grad_steps+1):
#                         logger.log('** Step ' + str(step) + ' **')

#                         """ -------------------- Sampling --------------------------"""

#                         logger.log("Obtaining samples...")
#                         time_env_sampling_start = time.time()

#                         # Meta-sampler's obtain_samples function now takes as input policy since we need trajectories for each policy
#                         initial_paths = [sampler.obtain_samples(
#                             policy=policy, log=True, log_prefix='Step_%d-' % step) for sampler in self.samplers]

#                         true_indices = []
#                         paths = OrderedDict()
#                         # len(self.envs) == len(initial_paths)
#                         # , Paths in enumerate(zip(*initial_paths)):
#                         for i in range(len(initial_paths[0])):
#                             index = np.random.choice(
#                                 list(range(len(initial_paths))), p=self.probs)
#                             paths[i] = initial_paths[index][i]
#                             true_indices.append(index)

#                         # list of 0's and 1's indicating which env
#                         true_indices = np.array(true_indices)
#                         list_sampling_time.append(
#                             time.time() - time_env_sampling_start)
#                         # (number of inner updates, meta_batch_size)
#                         all_paths.append(paths)

#                         """ ----------------- Processing Samples ---------------------"""

#                         logger.log("Processing samples...")
#                         time_proc_samples_start = time.time()
#                         samples_data = self.sample_processor.process_samples(
#                             paths, log='all', log_prefix='Step_%d-' % step)
#                         # (number of inner updates, meta_batch_size)
#                         all_samples_data.append(samples_data)

#                         # DEBUG
#                         # print("length of all_samples_data should be 40: {}".format(len(all_samples_data)))
#                         # print("all_samples_data[0] shape: {}".format(all_samples_data[0].shape))

#                         list_proc_samples_time.append(
#                             time.time() - time_proc_samples_start)

#                         self.log_diagnostics(
#                             sum(list(paths.values()), []), prefix='Step_%d-' % step)

#                         """ ------------------- Inner Policy Update --------------------"""
#                         if step < self.num_inner_grad_steps:
#                             inner_loop_losses = []

#                     # for algo in self.algos[:self.theta_count]: already looping over algos now so we don't need this
#                         time_inner_step_start = time.time()
#                         if step < self.num_inner_grad_steps:
#                             logger.log("Computing inner policy updates...")
#                             loss_list = algo._adapt(samples_data)
#                             inner_loop_losses.append(loss_list)

#                         if step == self.num_inner_grad_steps:
#                             indices = np.argmin(inner_loop_losses, axis=0)
#                             pred_indices = np.array(indices)
#                             clustering_score = np.abs(
#                                 np.mean(np.abs(true_indices - pred_indices)) - 0.5) * 2.0
#                             logger.logkv('Clustering Score', clustering_score)

# #                     algo_batches = [[] for _ in range(self.theta_count)]
# #                     for i in range(len(samples_data)):
# #                         index = indices[i]
# #                         algo_batches[index].append((i, samples_data[i]))

# #                     algo_all_samples.append(algo_batches)

#                         list_inner_step_time.append(
#                             time.time() - time_inner_step_start)
#                     total_inner_time = time.time() - start_total_inner_time

#                     time_maml_opt_start = time.time()
#                     """ ------------------ Outer Policy Update ---------------------"""

#                     logger.log("Optimizing policy...")
#                     # This needs to take all samples_data so that it can construct graph for meta-optimization.
#                     time_outer_step_start = time.time()
#                     # all_samples_index_data = [algo_batches[index]
#                     #                          for algo_batches in algo_all_samples]
#                     algo.optimize_policy(all_samples_data)

#                 """ ------------------- Logging Stuff --------------------------"""
#                 logger.logkv('Itr', itr)
#                 logger.logkv('n_timesteps', [
#                              sampler.total_timesteps_sampled for sampler in self.samplers])

#                 logger.logkv('Time-OuterStep', time.time() -
#                              time_outer_step_start)
#                 logger.logkv('Time-TotalInner', total_inner_time)
#                 logger.logkv('Time-InnerStep', np.sum(list_inner_step_time))
#                 logger.logkv('Time-SampleProc', np.sum(list_proc_samples_time))
#                 logger.logkv('Time-Sampling', np.sum(list_sampling_time))

#                 logger.logkv('Time', time.time() - start_time)
#                 logger.logkv('ItrTime', time.time() - itr_start_time)
#                 logger.logkv('Time-MAMLSteps', time.time() -
#                              time_maml_opt_start)

#                 logger.log("Saving snapshot...")
#                 params = self.get_itr_snapshot(itr)
#                 logger.save_itr_params(itr, params)
#                 logger.log("Saved")

#                 logger.dumpkvs()

#         logger.log("Training finished")
#         self.sess.close()

#     def get_itr_snapshot(self, itr):
#         """
#         Gets the current policy and env for storage
#         """
#         return dict(itr=itr, policies=self.policies, env=self.envs, baseline=self.baseline)

#     def log_diagnostics(self, paths, prefix):
#         # TODO: we aren't using it so far
#         #self.envs.log_diagnostics(paths, prefix)
#         # self.poli.log_diagnostics(paths, prefix)
#         self.baseline.log_diagnostics(paths, prefix)
