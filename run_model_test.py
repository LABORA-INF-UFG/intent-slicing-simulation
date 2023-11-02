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

from modelpack_v2.UserData import UserData
from modelpack_v2.ModelData import ModelData
from modelpack_v2.SliceData import SliceData
from modelpack_v2.modelOptimization import optimize

# Setting up the experiment

test_param = {
    "steps_per_trial": 2000, #2000,
    "total_trials": 50, #50,
    "initial_trial": 50, #46,
    "runs_per_agent": 1,
}

EMBB_USERS = 2 # Original = 4
URLLC_USERS = 2 # Original = 3
BE_USERS = 2 # Original = 3

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

# Bit generator for generating the amount of packets in each step
rng = np.random.default_rng(seed) if seed != -1 else np.random.default_rng()

# Setting up the simulation environment
env = Basestation(
    bs_name="test/{}/ws_{}/{}/".format(
        model,
        windows_size_obs,
        obs_space_mode,
    ),
    number_ues=EMBB_USERS + URLLC_USERS + BE_USERS,
    bandwidth=1e8, # Original = 1e8
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
env.reset(test_param["initial_trial"])

# Putting the simulation config data into the model data classes

data = ModelData(
    B=env.bandwidth,
    R=env.total_number_rbs,
    PS=env.packet_size,
    e=1e-5,
    l_max=env.buffer_max_lat,
    w_max=env.ues[0].windows_size
)
for s in env.slices:
    slice = SliceData(
        id=s.name,
        l_max=env.buffer_max_lat,
        hist_r=np.array([]),
        hist_d=np.array([]),
        hist_rcv=np.array([]),
        hist_buff=np.ndarray((0,env.buffer_max_lat+1)),
        hist_sent=np.ndarray((0,env.buffer_max_lat+1)),
        hist_part=np.array([])
    )
    for u in s.ues:
        ue = UserData(
            id=u.id,
            s=s.name,
            SE=u.se,
            b_max=u.max_packets_buffer
        )
        slice.addUser(ue)
    data.addSlice(slice)

def updateModelRequirements(env: Basestation, data: ModelData):
    for s in env.slice_requirements.keys():
        if s == "embb" or s == "urllc":
            data.slices[s].r_req = env.slice_requirements[s]["throughput"] * 1e3 * len(data.slices[s].users)
            data.slices[s].l_req = env.slice_requirements[s]["latency"]
            data.slices[s].p_req = env.slice_requirements[s]["pkt_loss"]
        elif s == "be":
            data.slices[s].g_req = env.slice_requirements[s]["long_term_pkt_thr"] * 1e3 * len(data.slices[s].users)
            data.slices[s].f_req = env.slice_requirements[s]["fifth_perc_pkt_thr"] * 1e3 * len(data.slices[s].users)

# Executing the experiment

print("\n############### Testing ###############")
for _ in tqdm(range(test_param["total_trials"] + 1 - test_param["initial_trial"]), leave=False, desc="Trials"):
    for _ in tqdm(range(test_param["steps_per_trial"]),leave=False, desc="Steps"):
        updateModelRequirements(env,data)

        # Updating model data before the simulation step
        for s in env.slices:    
            slice_d = 0
            slice_rcv = 0
            slice_part = 0
            slice_buff = np.zeros(data.l_max+1)
            
            for u in s.ues:
                d = u.dropped_pkts
                rcv = u.pkt_received
                part = u.partial_sent_pkts
                buff = u.buffer_array
                slice_d += d
                slice_rcv += rcv
                slice_part += part
                slice_buff += buff
            
            data.slices[s.name].updateHistBefStep(
                d=slice_d,
                rcv=slice_rcv,
                part=slice_part,
                buff=slice_buff
            )

        # Executing the optimization
        m, results = optimize(data=data, method="cplex", allocate_all_resources=False, verbose=False)
        if results.solver.termination_condition != "optimal":
            print("\nStep",data.n,"is unfeasible")
            exit()
        
        # Executing the simulation step with the optimal RBG scheduling
        scheduling = [m.R_s["be"].value, m.R_s["embb"].value, m.R_s["urllc"].value]
        env.step(scheduling, action_already_integer=True)

        # Updating model data after the simulation step
        for s in env.slices:
            slice_r = 0
            slice_sent = np.zeros(data.l_max+1)
            
            for u in s.ues:
                r = u.last_real_served_thr
                sent = u.sent_array
                slice_r += r
                slice_sent += sent
            
            data.slices[s.name].updateHistAftStep(
                    r=slice_r,
                    sent=slice_sent
                )
# Saving the model data
path = ("./hist/modeldata/trial{}/").format(test_param["initial_trial"])
try:
    os.makedirs(path)
except OSError:
    pass

np.savez_compressed(path + "modeldata", **data)