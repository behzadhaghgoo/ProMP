import tensorflow as tf
import numpy as np
import time
from meta_policy_search.utils import logger
import numpy as np
from sklearn.cluster import DBSCAN
from copy import deepcopy
from collections import OrderedDict
import pickle
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.meta_algos.trpo_maml import TRPOMAML


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

def arg_where(input_list, example):
    return [i for i in range(len(input_list)) if input_list[i] == example]

def calc_path_avg_return(path):
    undiscounted_returns = np.mean([sum(rollout["rewards"]) for rollout in path]) # Average returns
    return undiscounted_returns

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
            theta_count=1,
            multi_maml = False,
            phi_test = False,
            switch_thresh = False,
            mode_name = None,
            config=None,
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
        self.mini_saver = tf.train.Saver()
        self.num_envs = len(envs)
        self.meta_batch_size = self.samplers[0].meta_batch_size

        self.multi_maml = multi_maml
        self.phi_test = phi_test
        self.switch_thresh = switch_thresh
        self.mode_name = mode_name
        self.config = config
        assert len(samplers) == len(
            probs), "len(samplers) = {} != {} = len(probs)".format(len(samplers), len(probs))

        if sess is None:
            sess = tf.Session()
        self.sess = sess

        self.clusterer = DBSCAN(eps=0.1, min_samples=1)

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

        #######################################
        ###### Set experiment parameters ######
        #######################################
        multi_maml = self.multi_maml
        phi_test = self.phi_test
        switch_thresh = self.switch_thresh
        config = self.config
        load_checkpoint = False
        checkpoint_name = "KAML_with_late_theta_initialization_Iteration_675"
        do_it_with_return_please = True
        #######################################
        #######################################
        #######################################

        self.timer.start()

        with self.sess.as_default() as sess:
            # Load Checkpoint
            print("load_checkpoint is {}".format(load_checkpoint))
            if load_checkpoint:
                assert checkpoint_name, "Provide checkpoint name."
            if load_checkpoint:
                print("loading from checkpoint {}".format(checkpoint_name))
                saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_name))
                saver.restore(sess,checkpoint_name)

            list_sampling_time, list_inner_step_time, list_outer_step_time, list_proc_samples_time = [], [], [], []
            start_total_inner_time = time.time()
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables(
            ) if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

            start_time = time.time()

            num_thetas_used = 0 #  might need to be in iteration loop

            for itr in range(self.start_itr, self.n_itr):

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

                # shape : num_algos, num_inner_steps, num_tasks, ..
                all_algo_all_samples_data = [[]
                                             for _ in range(self.theta_count)]
                # shape : num_algos, num_tasks
                all_algo_inner_loop_losses = []
                all_algo_inner_loop_returns = []

                # For each theta in thetas, we obtain trajectories from the same tasks from both environments
                algo_samples_reward_data = []

                all_true_indices = []
                for task_ind in range(self.meta_batch_size):
                    index = np.random.choice(
                                list(range(self.num_envs)), p=self.probs)
                    all_true_indices.append(index)

                """self.algos is the global list of algos and every algo is denoted by its index there."""
                # new
                # first_theta = self.algos[0]
                first_theta = 0
                # thetas = {}
                # theta_vecs = {first_theta:}
                task_thetas = [first_theta for _ in range(self.meta_batch_size)]
                children = {first_theta: []}
                theta_depth = {first_theta: 1}
                max_depth = max([theta_depth[algo] for algo in theta_depth.keys()])
                all_algos_inner_loop_grads = []
                clusterer = DBSCAN(eps=0.3, min_samples=1)  

                # Refactored

                # Find the appropriate leaf theta for each task.
                algos_grads = {} # {algo_ind:[] for algo_ind, algo in enumerate(self.algos)}
                for depth in range(max_depth + 1):
                    # Sample path for each task and its theta
                    for algo_ind, algo in enumerate(self.algos):
                        if algo_ind not in task_thetas:
                            continue
                        step = 0
                        theta_task_inds = arg_where(task_thetas, algo_ind)
                        """-------------------- Sampling --------------------------"""
                        logger.log("Obtaining samples...")
                        true_indices = list(np.take(all_true_indices, theta_task_inds))
                        # Meta-sampler's obtain_samples function now takes as input policy since we need trajectories for each policy
                        algo.policy.switch_to_pre_update()  
                        initial_paths = [sampler.obtain_samples(
                            policy=algo.policy, log=True, log_prefix='Step_%d-' % step) for sampler in self.samplers]

                        print("len(initial_paths)", len(initial_paths))

                        # get paths no matter step == 0 or 1
                        paths = OrderedDict()
                        for task_ind in range(self.meta_batch_size):
                            index = true_indices[task_ind]
                            if task_ind in theta_task_inds:
                                paths[task_ind] = initial_paths[index][task_ind]

#                             paths = np.take(paths, theta_task_inds)
                        if step == self.num_inner_grad_steps:
                            algo_samples_reward_data.append(paths)

                        # list_sampling_time.append(
                        #     time.time() - time_env_sampling_start)

                        """----------------- Processing Samples ---------------------"""

                        logger.log("Processing samples...")
                        time_proc_samples_start = time.time()
                        samples_data = self.sample_processor.process_samples(
                            paths, log='all', log_prefix='Step_%d-' % step)

                        def calc_path_avg_return(path):
                            undiscounted_returns = np.mean([sum(rollout["rewards"]) for rollout in path]) # Average returns
                            return undiscounted_returns
                                
                        algo_returns = [calc_path_avg_return(paths[path]) for path in paths]
                        # print("len(algo_returns)", len(algo_returns))
                        all_algo_inner_loop_returns.append(algo_returns)

                        all_algo_all_samples_data[algo_ind].append(samples_data)

                        # list_proc_samples_time.append(
                        #     time.time() - time_proc_samples_start)

                        self.log_diagnostics(
                            sum(list(paths.values()), []), prefix='Step_%d-' % step)

                        """------------------- Inner Policy Update --------------------"""
                        if step == self.num_inner_grad_steps:
                            # In the last inner_grad_step, append inner loop losses of this algo to inner_loop_losses
                            all_algo_inner_loop_losses.append(
                                algo_inner_loop_losses)

                        time_inner_step_start = time.time()
                        if step < self.num_inner_grad_steps:
                            logger.log("Computing inner policy updates...")
                            algo_inner_loop_losses, _, algo_inner_loop_grads = algo._adapt(
                                samples_data)

                            algo_inner_loop_grads = np.array([np.concatenate(list([np.array(value).flatten() for value in d.values()])) for d in algo_inner_loop_grads])

                    # Calculate the gradient
                    # Update the mapping
                        if depth == max_depth:
                            algos_grads[algo_ind] = [algo_inner_loop_grads[i] for i in range(len(task_thetas)) if i in theta_task_inds]

                for algo_ind, algo in enumerate(self.algos):
                    if algo_ind not in task_thetas:
                        continue

                    clustering = clusterer.fit(algos_grads[algo_ind])
                    cluster_labels = clustering.labels_
                    print(cluster_labels, "\n\n\n\n\n\n\nNOT CLUSTERING\n\n\n\n\n\n\n")
                    if max(cluster_labels) > 0:
                        number_of_children = max(cluster_labels) + 1
                        print(cluster_labels, "\n\n\n\n\n\n\n{} Clusters".format(number_of_children))
                        children_algos = []
                        # sess = tf.InteractiveSession()
                        print("num_thetas_used = {}, len(self.algos) = {}".format(num_thetas_used, len(self.algos)))

                        
                        if number_of_children <= len(self.algos) - num_thetas_used:
                            
                        # Create children
                            for _ in range(number_of_children):
                                print(cluster_labels, "CLUSTERING\n\n\n\n\n\n\n")
                                if num_thetas_used + 1 < len(self.algos):
                                    print("Creating a child algo with theta number: {}".format(num_thetas_used + 1))
                                    
                                    new_child_algo = self.algos[num_thetas_used + 1]
                                    new_child_algo.policy.switch_to_pre_update() 
                                    num_thetas_used += 1
                                    parent_algo_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=algo.scope)
                                    child_algo_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=new_child_algo.scope)
                                    op_list = []
                                    for t, q in zip(child_algo_vars, parent_algo_vars):
                                        op_list.append(t.assign(q))

                                    update_target_op = tf.group(*op_list)
                                    sess.run(update_target_op)
    
    
    
                            print("task_thetas before: {}".format(task_thetas))
                            for i, ind in enumerate(theta_tasks):
                                task_thetas[ind] = num_thetas_used + cluster_labels[i]
                            print("task_thetas after: {}".format(task_thetas))
                        
                            children[algo_ind] = children_algos #[pickle.loads(pickle.dumps(algo, -1)) for _ in range(number_of_children)]
                            for cluster_ind, child_algo in enumerate(children[algo_ind]):
                                cluster_grads = algo_inner_loop_grads[np.argwhere(cluster_labels == cluster_ind)]
                                cluster_mean = np.squeeze(np.mean(cluster_grads, axis = 1))
                                assert len(cluster_mean.shape) == 1
                                assert cluster_mean.shape[0] > 1
                                theta_vecs[child_algo] = cluster_mean
                                theta_depth[child_algo] = theta_depth[algo_ind] + 1
                                children[child_algo] = []

                print("DOING OUTER LOOP")

                all_algo_all_samples_data = [[]
                                             for _ in range(self.theta_count)]
                # shape : num_algos, num_tasks
                all_algo_inner_loop_losses = []
                all_algo_inner_loop_returns = []

                # For each theta in thetas, we obtain trajectories from the same tasks from both environments
                algo_samples_reward_data = []
                
                for algo_ind, algo in enumerate(self.algos): 
                    
                    policy = algo.policy 
                    
                    for step in range(self.num_inner_grad_steps+1):
                        logger.log('** Step ' + str(step) + ' **')

                        """-------------------- Sampling --------------------------"""

                        logger.log("Obtaining samples...")
                        time_env_sampling_start = time.time()

                        theta_task_inds = arg_where(task_thetas, algo_ind) 
                        true_indices = list(np.take(all_true_indices, theta_task_inds))
                        
                        print("algo_ind: {}, theta tasks: {}".format(algo_ind, theta_task_inds)) 
                        if theta_task_inds == []:
                            print("theta_task_inds = []")
                            continue
                        # Meta-sampler's obtain_samples function now takes as input policy since we need trajectories for each policy
                        initial_paths = [sampler.obtain_samples(
                            policy=policy, log=True, log_prefix='Step_%d-' % step) for sampler in self.samplers]

                        print("len(initial_paths)", len(initial_paths))
                        #assert 1 == 2
                        # get paths no matter step == 0 or 1
                        paths = OrderedDict()

                        for task_ind in theta_task_inds:
                            index = true_indices[task_ind]
                            paths[task_ind] = initial_paths[index][task_ind]

                        if step == self.num_inner_grad_steps:
                            algo_samples_reward_data.append(paths)

                       #  list_sampling_time.append(
                       #      time.time() - time_env_sampling_start)

                        """----------------- Processing Samples ---------------------"""

                        logger.log("Processing samples...")
                        time_proc_samples_start = time.time()
                        samples_data = self.sample_processor.process_samples(
                            paths, log='all', log_prefix='Step_%d-' % step)
                        print("len(samples_data)", len(samples_data))
                        # (number of inner updates, meta_batch_size)


                        def calc_path_avg_return(path):
                            undiscounted_returns = np.mean([sum(rollout["rewards"]) for rollout in path]) # Average returns
                            return undiscounted_returns

                        if step == self.num_inner_grad_steps:
                            algo_returns = [calc_path_avg_return(paths[path]) for path in paths]
                            print("len(algo_returns)", len(algo_returns))
                            all_algo_inner_loop_returns.append(algo_returns)

                        all_algo_all_samples_data[algo_ind].append(samples_data)

                        # list_proc_samples_time.append(
                        #     time.time() - time_proc_samples_start)

                        self.log_diagnostics(
                            sum(list(paths.values()), []), prefix='Step_%d-' % step)

                        """------------------- Inner Policy Update --------------------"""
                        if step == self.num_inner_grad_steps:
                            # In the last inner_grad_step, append inner loop losses of this algo to inner_loop_losses
                            all_algo_inner_loop_losses.append(
                                algo_inner_loop_losses)

                        time_inner_step_start = time.time()
                        if step < self.num_inner_grad_steps:
                            logger.log("Computing inner policy updates...")
                            algo_inner_loop_losses, _, algo_inner_loop_grads = algo._adapt(
                                samples_data)


                            list_inner_step_time.append(
                                time.time() - time_inner_step_start)
                        total_inner_time = time.time() - start_total_inner_time

                        time_maml_opt_start = time.time()
                        """------------------ Outer Policy Update ---------------------"""

                        logger.log("Optimizing policy...")

                        time_outer_step_start = time.time()

                        # all_algo_all_samples_data = np.array(all_algo_all_samples_data)
                        # all_algo_inner_loop_losses = np.array(
                        #     all_algo_inner_loop_losses)
                        true_indices = np.array(true_indices)

#                         if do_it_with_return_please:
#                             print("wise choice")
#                             if (multi_maml or phi_test) and itr < switch_thresh:
#                                 which_algo = true_indices
#                             else:
#                                 which_algo = np.argmax(
#                                     all_algo_inner_loop_returns, axis=0)  # length num_tasks
#                         else:
#                             print("think twice")
#                             if (multi_maml or phi_test) and itr < switch_thresh:
#                                 which_algo = true_indices
#                             else:
#                                 which_algo = np.argmin(
#                                     all_algo_inner_loop_losses, axis=0)  # length num_tasks

#                         # print("which_algo shape: ", which_algo.shape)

                # For each algo, do outer update
                relevant_paths = OrderedDict()
                count = 0
                print("task thetas now: ", list(set(task_thetas)))
                for algo_ind, algo in enumerate(self.algos):
                    # Get all indices of data from tasks that were assigned to this algo
                    
                    if algo_ind not in task_thetas:
                        continue 

                    print("arg_where(task_thetas, algo)", arg_where(task_thetas, algo_ind))
                    theta_task_inds = arg_where(task_thetas, algo_ind)
                    # print("which_algo = {}, theta_task_inds = {}".format(which_algo, theta_task_inds))
                    relevant_datalgo_indices = theta_task_inds #(which_algo == algo_ind)
                    # print("relevant_datalgo_indices", relevant_datalgo_indices.shape)
                    relevant_datalgo_indices = np.nonzero(
                        relevant_datalgo_indices)[0]
                    # print("relevant_datalgo_indices", relevant_datalgo_indices.shape)
                    print(
                        "all_algo_all_samples_data[algo_ind, :, relevant_datalgo_indices]")

                    print("np.array(all_algo_all_samples_data).shape", np.array(all_algo_all_samples_data).shape)

                    # print("all_algo_all_samples_data.shape", all_algo_all_samples_data.shape)
                    for i in range(len(all_algo_all_samples_data)):
                        
                        print("len(all_algo_all_samples_data[0])", len(all_algo_all_samples_data[i]))
                    # Fill the batch to make the shape right.
                    

                    x = (np.array(all_algo_all_samples_data[algo_ind])[:, list(relevant_datalgo_indices)])  # 21 x 2

                    path_list = algo_samples_reward_data[algo_ind]

                    for index in relevant_datalgo_indices:
                        path = path_list[index]
                        relevant_paths[count] = path
                        count += 1

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
                    self.algos[algo_ind].optimize_policy(new_x.T)

                    self.sample_processor._helper(relevant_paths, log='reward', log_prefix='Step_2-')

#                             clustering_score = np.abs(
#                                 np.mean(np.abs(true_indices - which_algo)) - 0.5) * 2.0
                    # logger.logkv('Clustering Score', clustering_score)

                    """------------------- Logging Stuff --------------------------"""
                    logger.logkv('Itr', itr)
                    logger.logkv('n_timesteps', [
                                 sampler.total_timesteps_sampled for sampler in self.samplers])

                    logger.logkv('Time-OuterStep', time.time() -
                                 time_outer_step_start)
                    logger.logkv('Time-TotalInner', total_inner_time)
                    logger.logkv('Time-InnerStep', np.sum(list_inner_step_time))
                    # logger.logkv('Time-SampleProc', np.sum(list_proc_samples_time))
                    # logger.logkv('Time-Sampling', np.sum(list_sampling_time))

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
