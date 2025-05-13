import argparse
import configparser

import data_preprocess as dp
import codeco_rl_env_v2 as cev2
import kubernetes_emmulator as ke
import training_controller

def init_config(path_config):
    # Create a ConfigParser object
    config = configparser.ConfigParser()

    # Read the .ini file, we encourage the user to add more configuration parameters if desired
    config.read(path_config)
    config_dqn = {
        'lr': config.getfloat('Train_config', 'lr'),
        'e_fraction': config.getfloat('Dqn_config', 'e_fraction'),
        'buffer_size': config.getint('Dqn_config', 'buffer_size'),
        'tau': config.getfloat('Dqn_config', 'tau'),
        'gamma': config.getfloat('Dqn_config', 'gamma')
    }

    config_learn = {
        'steps': config.getint('Train_config', 'steps'),
        'progress_bar': config.getboolean('Train_config', 'progress_bar'),
        'log_interval': config.getint('Train_config', 'log_interval'),
        'episode_length': config.getint('Train_config', 'episode_length'),
        'npredictions': config.getint('Train_config','npredictions')
    }

    return config_dqn, config_learn


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="A script that accepts command-line arguments.")
    parser.add_argument("--node_path", default="data/node_info.csv", help="Path to node config info")
    parser.add_argument("--data_path", default="data/test_t.csv", help="Path to data")
    parser.add_argument("--save_models_path", default="3node4", help="Path to save models")
    parser.add_argument("--path_saving_model", default="tests", help="Path of model to use")
    args = parser.parse_args()

    node_path = args.node_path
    data_path = args.data_path
    save_models_path = args.save_models_path
    path_model_use = args.path_saving_model
    
    # Initializing data structures with necessary data
    config_train, config_learn = init_config("data/config_train.ini")
    df_node, df_pods_queue, nodes_info = dp.read_and_create(node_path, data_path)
    
    episode_l = config_learn['episode_length']
    df_training = df_pods_queue[:episode_l]
    env_config={"df_node":df_node, "df_data":df_training, "nodes_info":nodes_info,
                "w_ld_cpu": 5, "w_ld_ram": 5, "w_eo": 0, "w_tc": 0, "w_ec": 5,}
        
    emmulator = ke.KubernetesEmulator(env_config)
    env_config["emmulator"] = emmulator
    env_config["padding"] = True

    print("Initial configuration done")
    
    env = cev2.CodecoEnvV2(env_config)
    trainer = training_controller.Codeco_trainer(env)  

    trainer.train_ppo(config_train, config_learn, path_model_use)


