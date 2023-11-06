import os

import joblib
import numpy as np
from stable_baselines3 import SAC, TD3, PPO, DDPG
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

from baselines import BaselineAgent
from basestation import Basestation
from callbacks import ProgressBarManager

test_param = {
    "steps_per_trial": 2000, #2000,
    "total_trials": 50, #50,
    "initial_trial": 50, #46,
    "runs_per_agent": 1,
}

# Create environment
EMBB_USERS = 4 # Original = 4
URLLC_USERS = 3 # Original = 3
BE_USERS = 3 # Original = 3

traffic_types = np.concatenate(
    (
        np.repeat(["embb"], EMBB_USERS),
        np.repeat(["urllc"], URLLC_USERS),
        np.repeat(["be"], BE_USERS),
    ),
    axis=None,
)

traffic_throughputs = {
    "light": {
        "embb": 15, # Mbps = Kb/ms = Kb/step
        "urllc": 1,
        "be": 15,
    },
    "moderate": {
        "embb": 25,
        "urllc": 5,
        "be": 25,
    },
}
slice_requirements_traffics = {
    "light": {
        "embb": {"throughput": 10, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 1, "latency": 1, "pkt_loss": 1e-5},
        "be": {"long_term_pkt_thr": 5, "fifth_perc_pkt_thr": 2},
    },
    "moderate": {
        "embb": {"throughput": 20, "latency": 20, "pkt_loss": 0.2},
        "urllc": {"throughput": 5, "latency": 1, "pkt_loss": 1e-5},
        "be": {"long_term_pkt_thr": 10, "fifth_perc_pkt_thr": 5},
    },
}

#models = ["intentless", "colran", "sac"]
models = ["sac"]
obs_space_modes = ["partial"] # , "full"]
windows_sizes = [1]  # , 50, 100] # Window for calculating the moving mean of metrics (not used)
seed = 100 
n_eval_episodes = 5  # default is 5
eval_freq = 10000  # default is 10000
test_model = "best"  # or last


# Instantiate the agent
def create_agent(
    type: str,
    env: VecNormalize,
    mode: str,
    obs_space_mode: str,
    windows_size_obs: int,
    test_model: str = "best",
):
    def optimized_hyperparameters(model: str, obs_space: str):
        hyperparameters = joblib.load(
            "hyperparameter_opt/{}_{}_ws{}.pkl".format(
                model, obs_space, windows_size_obs
            )
        ).best_params
        net_arch = {
            "small": [64, 64],
            "medium": [256, 256],
            "big": [400, 300],
        }[hyperparameters["net_arch"]]
        hyperparameters["policy_kwargs"] = dict(net_arch=net_arch)
        hyperparameters.pop("net_arch")
        hyperparameters["target_entropy"] = "auto"
        hyperparameters["ent_coef"] = "auto"
        hyperparameters["gradient_steps"] = hyperparameters["train_freq"]

        return hyperparameters

    if mode == "train":
        if type == "sac":
            hyperparameters = optimized_hyperparameters(type, obs_space_mode)
            return SAC(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./tensorboard-logs/",
                **hyperparameters,
                seed=seed,
            )
        elif type == "td3":
            hyperparameters = optimized_hyperparameters(type, obs_space_mode)
            return TD3(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./tensorboard-logs/",
                **hyperparameters,
                seed=seed,
            )
        elif type == "intentless":
            return DDPG(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./tensorboard-logs/",
                seed=seed,
            )
        elif type == "colran":
            return PPO(
                "MlpPolicy",
                env,
                verbose=0,
                tensorboard_log="./tensorboard-logs/",
                seed=seed,
            )
    elif mode == "test":
        path = (
            "./agents/best_{}_{}_ws{}/best_model".format(
                type, obs_space_mode, windows_size_obs
            )
            if test_model == "best"
            else "./agents/{}_{}_ws{}".format(type, obs_space_mode, windows_size_obs)
        )
        if type == "sac":
            return SAC.load(
                path,
                None,
                verbose=0,
            )
        elif type == "td3":
            return TD3.load(
                path,
                None,
                verbose=0,
            )
        elif type == "intentless":
            return DDPG.load(
                path,
                None,
                verbose=0,
            )
        elif type == "colran":
            return PPO.load(
                path,
                None,
                verbose=0,
            )
        elif type == "mt":
            return BaselineAgent("mt")
        elif type == "pf":
            return BaselineAgent("pf")
        elif type == "rr":
            return BaselineAgent("rr")


# Test
print("\n############### Testing ###############")
#models_test = np.append(models, ["mt", "rr", "pf"])
models_test = models
for windows_size_obs in tqdm(windows_sizes, desc="Windows size", leave=False):
    for obs_space_mode in tqdm(obs_space_modes, desc="Obs. Space mode", leave=False):
        for model in tqdm(models_test, desc="Models", leave=False):
            rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()
            env = Basestation(
                bs_name="test/{}/ws_{}/{}/".format(
                    model,
                    windows_size_obs,
                    obs_space_mode,
                ),
                bandwidth=1e8, # Original = 1e8
                total_number_rbs = 100, # Original = 17
                number_ues=EMBB_USERS + URLLC_USERS + BE_USERS,
                max_number_steps=test_param["steps_per_trial"],
                max_number_trials=test_param["total_trials"],
                traffic_types=traffic_types,
                traffic_throughputs=traffic_throughputs,
                slice_requirements_traffics=slice_requirements_traffics,
                windows_size_obs=windows_size_obs,
                obs_space_mode=obs_space_mode,
                rng=rng,
                plots=True,
                save_hist=True,
                baseline=False,
            )

            if model in models:
                dir_vec_models = "./vecnormalize_models"
                dir_vec_file = dir_vec_models + "/{}_{}_ws{}.pkl".format(
                    model, obs_space_mode, windows_size_obs
                )
                env = Monitor(env) # Stable baselines wrapper
                dict_reset = {"initial_trial": test_param["initial_trial"]}
                obs, _ = env.reset(**dict_reset)
                obs = [obs]
                env = DummyVecEnv([lambda: env])
                env = VecNormalize.load(dir_vec_file, env) # env is normalized
                env.training = False
                env.norm_reward = False
            elif not (model in models):
                obs, _ = env.reset(test_param["initial_trial"])
            agent = create_agent(
                model, env, "test", obs_space_mode, windows_size_obs, test_model
            )
            agent.set_random_seed(seed)
            for _ in tqdm(
                range(test_param["total_trials"] + 1 - test_param["initial_trial"]),
                leave=False,
                desc="Trials",
            ):
                for _ in tqdm(
                    range(test_param["steps_per_trial"]),
                    leave=False,
                    desc="Steps",
                ):
                    action, _states = (
                        agent.predict(obs, deterministic=True)
                        if model in models
                        else agent.predict(obs)
                    )
                    step = env.step(action)
                    if (len(step) == 4):
                        obs, rewards, dones, info = step
                    else:
                        obs, rewards, dones, _, info = step
                if model not in models:
                    env.reset()
