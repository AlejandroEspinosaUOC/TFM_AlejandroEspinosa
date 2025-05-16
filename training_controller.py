import time

import torch as th
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

##### CODE FROM OLDER PROJECTS #####

class Codeco_trainer():
    def __init__(self, env):
        self.env_to_train = env

    def train_ppo(self, configdqn, configlearn,path,config=None):
        run = wandb.init(
            project="codecov2",
            #config=config,
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        )
        env = self.env_to_train
        obs, _ = env.reset() 
        run = wandb.init(config=config)

        policy_kwargs = dict(
            activation_fn=th.nn.LeakyReLU,
            net_arch= dict(pi=[128, 64, 32, 16, 16, 16], vf=[128, 64, 32, 16, 16, 16])
            
        )
        
        if (th.cuda.is_available()):
            print("Using GPU")
            if self.env_to_train.padding:
                # Masking PPO
                model = MaskablePPO("MlpPolicy", gamma=0.95, ent_coef= 0.05, clip_range=0.15, batch_size=64, env=env, policy_kwargs=policy_kwargs,verbose=2, learning_rate=configdqn['lr'], tensorboard_log="logs_test_tensorboard/", device='cuda')
                print("training!")
            else:
                model = PPO("MlpPolicy", gamma=0.8, ent_coef= 0.1, clip_range=0.2, batch_size=64, env=env, policy_kwargs=policy_kwargs,verbose=2, learning_rate=configdqn['lr'], tensorboard_log="logs_test_tensorboard/", device='cuda')
        else:
            print("Using CPU")
            if self.env_to_train.padding:
                model = MaskablePPO("MlpPolicy", gamma=0.95, ent_coef= 0.05, clip_range=0.15, batch_size=64, env=env, policy_kwargs=policy_kwargs,verbose=2, learning_rate=configdqn['lr'], tensorboard_log="logs_test_tensorboard/", device='cpu')
            else:
                model = PPO("MlpPolicy", ent_coef= 0.03, batch_size=config.batch_size, clip_range=0.3, env=env, policy_kwargs=policy_kwargs,verbose=2, learning_rate=config.lr, tensorboard_log="logs_test_tensorboard/", device='cpu')
        
        model.learn(total_timesteps=configlearn['steps'], progress_bar=configlearn['progress_bar'], log_interval=configlearn['log_interval'], callback=WandbCallback())
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        model.save(path+timestr)
        
        run.finish()
        print("Training finished")
        print("Model saved in:", path+timestr)