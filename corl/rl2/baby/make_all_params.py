import argparse
import datetime
import os
import json
import pathlib

import dateutil.tz
import joblib
import tensorflow as tf



if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Play a pickled policy.')
    parser.add_argument('variant_index', metavar='variant_index', type=int,
                    help='The index of variants to use for experiment')
    parser.add_argument('--dir', metavar='dir', type=str,
                    help='The path of the folder that contains pkl files',
                    default=None, required=False)
    parser.add_argument('--pkl', metavar='pkl', type=str,
                    help='The path of the pkl file',
                    default=None, required=False)
    parser.add_argument('--config', metavar='config', type=str,
                    help='The path to the config file',
                    default=None, required=False)
    parser.add_argument('--itr', metavar='itr', type=int,
                    help='The start itr of the resuming experiment',
                    default=0, required=False)
    args = parser.parse_args()

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    idx = args.variant_index
    pkl = args.pkl
    folder = args.dir
    config_file = args.config
    itr = args.itr

    from os import listdir
    from os.path import isfile
    import os.path
    pkls = [file for file in listdir(folder) if '.pkl' in file]

    if not config_file:
        config_file = './corl/rl2/configs/baby_mode_config{}.json'.format(idx)

    if pkl:
        raise NotImplementedError
        pkl_path = pathlib.Path(pkl)
        exp_name = pkl_path.parts[-2]
        pkl_itr = int(pkl_path.parts[-1].split('_')[-1].split('.')[0])
        with tf.Session() as sess:
            with open(pkl, 'rb') as file:
                experiment = joblib.load(file)
            logger.configure(dir='./data/rl2/eval_{}'.format(exp_name), format_strs=['stdout', 'log', 'csv', 'json', 'tensorboard'],
                     snapshot_mode='all',)
            config = json.load(open(config_file, 'r'))
            json.dump(config, open('./data/rl2/eval_{}/params.json'.format(exp_name), 'w'))
            rl2_eval(experiment, config, sess, pkl_itr, pkl)
    elif folder:
        print('Found {} pkls...'.format(len(pkls)))
        all_params = {}
        for p in sorted(pkls):
            print('Processing {}...'.format(p))
            exp_path = pathlib.Path(folder)
            exp_name = exp_path.parts[-1]
            pkl_itr = int(p.split('_')[-1].split('.')[0])

            with tf.Graph().as_default():
                with tf.Session() as sess:
                    with open(os.path.join(folder, p), 'rb') as file:
                        experiment = joblib.load(file)
                        params = experiment['policy'].state['network_params']
                        all_params[pkl_itr] = params

                sess.close()
            tf.reset_default_graph()

        eval_path = pathlib.Path('./data/rl2/eval_{}'.format(exp_name))
        eval_path.mkdir(exist_ok=True)
        joblib.dump(all_params, './data/rl2/eval_{}/all_params.pkl'.format(exp_name))
    else:
        print('Please provide a pkl file')
