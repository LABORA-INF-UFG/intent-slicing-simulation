import os

import numpy as np
from tqdm import tqdm
import pickle

from baselines import BaselineAgent
from basestation import Basestation
from callbacks import ProgressBarManager

from modelpack_v3.UserData import UserData
from modelpack_v3.ModelData import ModelData
from modelpack_v3.SliceData import SliceData
from modelpack_v3.modelOptimization import optimize

# Setting up the experiment

test_param = {
    "steps_per_trial": 200, #2000,
    "total_trials": 50, #50,
    "initial_trial": 50, #46,
    "runs_per_agent": 1,
}

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
        "embb": 15, # Original = 15
        "urllc": 1, # Original = 1
        "be": 15, # Original = 15
    },
    "moderate": {
        "embb": 25, # Original = 25
        "urllc": 5, # Original = 5
        "be": 25, # Original = 25
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
# slice_requirements_traffics = {
#     "light": {
#         "embb": {"throughput": 10, "latency": 50, "pkt_loss": 0.5},
#         "urllc": {"throughput": 1, "latency": 1, "pkt_loss": 1e-5},
#         "be": {"long_term_pkt_thr": 5, "fifth_perc_pkt_thr": 2},
#     },
#     "moderate": {
#         "embb": {"throughput": 20, "latency": 50, "pkt_loss": 0.5},
#         "urllc": {"throughput": 5, "latency": 1, "pkt_loss": 1e-5},
#         "be": {"long_term_pkt_thr": 10, "fifth_perc_pkt_thr": 5},
#     },
# }

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
    total_number_rbs = 100, # Original = 17
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
    e=1e-6,
    b_max=env.max_packets_buffer,
    l_max=env.buffer_max_lat,
    w_max=env.ues[0].windows_size
)
for s in env.slices:
    data.addSlice(s.name)
for u in env.ues:
    data.addUser(id=u.id, s=u.traffic_type, SE=u.se)
data.associateUsersToSlices()

# Updates model requirements (may change in each step)
def updateModelRequirements(env: Basestation, data: ModelData):
    for u in data.users.values():
        if u.s == "embb" or u.s == "urllc":
            u.updateRequirements(
                r_req=env.slice_requirements[u.s]["throughput"] * 1e3, # Converting to bits
                l_req=env.slice_requirements[u.s]["latency"],
                p_req=env.slice_requirements[u.s]["pkt_loss"]
            )
        elif u.s == "be":
            u.updateRequirements(
                g_req=env.slice_requirements[u.s]["long_term_pkt_thr"] * 1e3, # Converting to bits
                f_req=env.slice_requirements[u.s]["fifth_perc_pkt_thr"] * 1e3 # Converting to bits
            )

# Updates model data before the simulation step
def updateModelDataBefStep(env: Basestation, data: ModelData):
    for u in env.ues:    
        data.users[u.id].updateHistBefStep(
            d=u.dropped_pkts,
            rcv=u.pkt_received,
            part=u.partial_sent_pkts,
            buff=u.buffer_array
        )
            
    for s in env.slices:    
        data.slices[s.name].updateHistBefStep(data.n)

# Updates model data after the simulation step
def updateModelDataAftStep(env: Basestation, data: ModelData):
    for u in env.ues:    
        data.users[u.id].updateHistAftStep(
            r = u.last_real_served_thr,
            sent = u.sent_array
        )
            
    for s in env.slices:    
        data.slices[s.name].updateHistAftStep(data.n)

# Extracts results from the optimization model and save them in the model data
def saveResults(data: ModelData, m):
    rrbs_per_slice = {
        # "embb": m.a_s["embb"].value * len(data.slices["embb"].users),
        # "be": m.a_s["be"].value * len(data.slices["be"].users),
        # "urllc" : m.a_s["urllc"].value * len(data.slices["urllc"].users)
        "embb": m.R_s["embb"].value,
        "be": m.R_s["be"].value,
        "urllc" : m.R_s["urllc"].value
    }
    
    data.saveResults(rrbs_per_slice)

# Updates the Round-Robin prioritization for UEs in a slice
# The first element is the prior UE index
def updateRRPrioritization(env: Basestation, data: ModelData):
    for s in env.slices:
        prior = np.arange(len(s.ues))[::-1] # Start prioritizing higher indexes
        prior = np.roll(prior, s.rr_index)
        for i in range(len(prior)):
            prior[i] = list(data.slices[s.name].users.keys())[prior[i]]
        data.slices[s.name].rr_prioritization = prior

# Executing the experiment
print("\n############### Testing ###############")
for _ in tqdm(range(test_param["total_trials"] + 1 - test_param["initial_trial"]), leave=False, desc="Trials"):
    for _ in tqdm(range(test_param["steps_per_trial"]),leave=False, desc="Steps"):
        # Updating model data
        updateModelRequirements(env, data)
        updateRRPrioritization(env, data)
        updateModelDataBefStep(env, data)

        # Executing the optimization
        m, results = optimize(data=data, method="cplex", allocate_all_resources=False, verbose=False)
        if results.solver.termination_condition != "optimal":
            print("\nStep",data.n,"is unfeasible")
            env.save_hist()
            exit()

        # Extracting the optimal RBG scheduling from the solution
        # be_resources = m.a_s["be"].value * len(data.slices["embb"].users)
        # embb_resources = m.a_s["embb"].value * len(data.slices["be"].users)
        # urllc_resources = m.a_s["urllc"].value * len(data.slices["urllc"].users)
        be_resources = m.R_s["be"].value
        embb_resources = m.R_s["embb"].value
        urllc_resources = m.R_s["urllc"].value
        
        # Saving results
        saveResults(data, m)

        # Executing the simulation step with the optimal RBG scheduling
        scheduling = [be_resources, embb_resources, urllc_resources]
        env.step(scheduling, action_already_integer=True)

        # Updating model data
        updateModelDataAftStep(env, data)
        data.advanceStep()

# Saving the model data
path = ("./hist/modeldata/trial{}/").format(test_param["initial_trial"])
os.makedirs(path, exist_ok=True)
with open(path+"modeldata.pickle", "wb") as model_data_file:
    pickle.dump(data, model_data_file)