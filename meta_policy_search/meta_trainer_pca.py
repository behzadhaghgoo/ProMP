import tensorflow as tf
import numpy as np
import time
from meta_policy_search.utils import logger
import numpy as np
import pickle 


from collections import OrderedDict

# Grad Clustering
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.cluster import KMeans


import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

do_it_with_return_please = True


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
        labels = [] 

        #######################################
        #######################################
        #######################################

        load_checkpoint = False
        print("load_checkpoint is {}".format(load_checkpoint))

        #######################################
        ############# Loader Code #############
        #######################################

        checkpoint_name = "KAML_with_late_theta_initialization_Iteration_675"
        if load_checkpoint:
            assert checkpoint_name, "Provide checkpoint name."
        else:
            print("not loading checkpoint")
            
        #######################################
        #######################################
        #######################################

        with self.sess.as_default() as sess:

            if load_checkpoint:
                print("loading from checkpoint {}".format(checkpoint_name))
                saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint_name))
                saver.restore(sess,checkpoint_name)

                
            print("starting")
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
                all_algo_inner_loop_returns = []

                true_indices = []
                for task_ind in range(self.meta_batch_size):
                    index = np.random.choice(
                        list(range(self.num_envs)), p=self.probs)
                    true_indices.append(index)
                print("true_indices: ", true_indices)

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

                        
                        def calc_path_avg_return(path):
#                             average_discounted_return = np.mean(
#                                 [path["returns"][0] for path in paths])
#                             print("paths[0]", paths[0])
#                             print('path["rewards"]', paths[0][0]["rewards"])
#                             print("len(path)", len(path))
#                             print("len(rollout)",len(path[0]))
#                             print("len(rollout)['rewards']",path[0]["rewards"])
        
                            undiscounted_returns = np.mean([sum(rollout["rewards"]) for rollout in path]) # Average returns
                            return undiscounted_returns
                        
                        if step == self.num_inner_grad_steps:
#                             print("path['returns']", paths[0]["returns"])
#                             print("len(paths)", len(paths))
#                             for i, path in enumerate(paths):
#                                 print("path {} = {}".format(i, path))
                            algo_returns = [calc_path_avg_return(paths[path]) for path in paths]
                            print("len(algo_returns)", len(algo_returns))
                            all_algo_inner_loop_returns.append(algo_returns)


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
                            #print("len(algo_inner_loop_grads)", len(algo_inner_loop_grads))
                            #print("algo_inner_loop_grads[0] shape: ", algo_inner_loop_grads[0].shape)
                            if itr % 10 == 0: 
                                pickle.dump( algo_inner_loop_grads, open("nk_grads_iter_{}_.p".format(itr), "wb") )
                                print("\n\n\n\n\n\n")
                                print("saving true_indices: ", true_indices)
                                print("\n\n\n\n\n\n")
                                pickle.dump( true_indices, open( "nk_true_indices_iter_{}_.p".format(itr), "wb" ) ) 

                        time_inner_step_start = time.time()
                        if step < self.num_inner_grad_steps:
                            logger.log("Computing inner policy updates...")
                            algo_inner_loop_losses, _, algo_inner_loop_grads = algo._adapt(
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

                if do_it_with_return_please:
                    print("wise choice")
                    if (multi_maml or phi_test) and itr < switch_thresh:
                        which_algo = true_indices
                    else:
                        which_algo = np.argmax(
                            all_algo_inner_loop_returns, axis=0)  # length num_tasks
                else:
                    print("think twice")
                    if (multi_maml or phi_test) and itr < switch_thresh:
                        which_algo = true_indices
                    else:
                        which_algo = np.argmin(
                            all_algo_inner_loop_losses, axis=0)  # length num_tasks

                # print("which_algo shape: ", which_algo.shape)

                # For each algo, do outer update
                relevant_paths = OrderedDict()
                count = 0
                
        
                algo_grads = {} 
            
                for a_ind, algo in enumerate(self.algos[:self.theta_count]):
                    # Get all indices of data from tasks that were assigned to this algo
#                     a_ind = a_ind_p - 1
                    
                    print("a_ind", a_ind)
                        
                    start_clustering_at = 10 
                    if itr > start_clustering_at:
                        if a_ind != 0: 
                            relevant_data_indices = (labels == a_ind - 1)
                        else:
                            relevant_data_indices = list(range(self.meta_batch_size)) 
                    else: 
                        relevant_data_indices = len(which_algo) * [1]
                    print("relevant_data_indices", relevant_data_indices)
                    
                    # print("relevant_data_indices", relevant_data_indices.shape)
                    relevant_data_indices = np.nonzero(
                        relevant_data_indices)[0]
                    
                    if len(relevant_data_indices) == 0:
                        continue
                    # print("relevant_data_indices", relevant_data_indices.shape)
                    print(
                        "all_algo_all_samples_data[a_ind, :, relevant_data_indices]")

                    # Fill the batch to make the shape right.
                    x = (all_algo_all_samples_data[a_ind, :, list(
                        relevant_data_indices)])  # 21 x 2

#                     print(algo_samples_reward_data)
                    path_list = algo_samples_reward_data[a_ind]

                    for index in relevant_data_indices:
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
                    if a_ind != 0: 
                        print("optimizing policy for algo: {}".format(a_ind))
                        algo.optimize_policy(new_x.T)
                        
                    if a_ind == 0: 
                        gradients = algo.compute_outer_gradients(new_x.T) 
                        algo_grads[a_ind] = gradients 
                    
                
                    ##########
                    
                clusterer = KMeans(n_clusters=2)#self.theta_count)
                  # np.array([np.concatenate(list([np.array(value).flatten() for value in d.values()])) for d in algo_grads[0]])
                
                start_clustering_at = 10 
                if itr >= start_clustering_at:
                    
                    sample_grads = np.array(algo_grads[0])
                    
                    if itr == start_clustering_at:
                        print("itr: {}, fitting PCA".format(itr))
                        pca = PCA(n_components = 2)
                        pca_result = pca.fit_transform(sample_grads)

                    t_vecs = pca.transform(sample_grads)
                    
                    if itr % 10 == 0: 
                        pickle.dump( sample_grads, open("nk_outer_grads_iter_{}_.p".format(itr), "wb") )

                    df = pd.DataFrame()
                    df["x"] = t_vecs[:,0]
                    df["y"] = t_vecs[:,1]

                    if itr == start_clustering_at:
                        print("fitting Kmeans")
                        clustering = clusterer.fit(t_vecs)
                        labels = clustering.labels_

                    labels = clustering.predict(t_vecs)    

                    print("clustering.labels_", labels)

                        # Plot the grads
                    if itr % 10 == 0:
#                         plt.xlim(-5,5)
#                         plt.ylim(-3,3)
                        sns.scatterplot(
                        x=df['x'], y=df['y'],
                        hue=labels,
                        palette=[color for i, color in enumerate(sns.color_palette("hls", max(clustering.labels_) - min(clustering.labels_) + 1)) if i in labels],
                        data=df,
                        legend="full",
                        alpha=1
                        )
                        plt.savefig("grads_itr_{}.png".format(itr))
                        plt.close()
                        ##########
                    
                    
                    

                self.sample_processor._helper(relevant_paths, log='reward', log_prefix='Step_2-')

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
