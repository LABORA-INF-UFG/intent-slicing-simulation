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
    "total_trials": 4, #50,
    "initial_trial": 4, #46,
    "runs_per_agent": 1,
}

# Create environment
traffic_types = np.concatenate(
    (
        np.repeat(["embb"], 4), # 4 EMBB UEs
        np.repeat(["urllc"], 3), # 3 URLLC UEs
        np.repeat(["be"], 3), # 3 BE UEs
    ),
    axis=None,
)
traffic_throughputs = {
    "light": {
        "embb": 15,
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

model = "optimal"
obs_space_mode = "partial"
windows_size_obs = 1
seed = 100 

print("\n############### Testing ###############")

# Bit generator for generating the amount of packets in each step
rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()
env = Basestation(
    bs_name="test/{}/ws_{}/{}/".format(
        model,
        windows_size_obs,
        obs_space_mode,
    ),
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

for _ in tqdm(range(test_param["total_trials"] + 1 - test_param["initial_trial"]), leave=False, desc="Trials"):
    for _ in tqdm(range(test_param["steps_per_trial"]),leave=False, desc="Steps"):
        # INSERT OPTIMAL CALCULATION HERE
        
        scheduling = [1,3,5] # Get optimization result here
        step = env.step(scheduling, action_already_integer=True)