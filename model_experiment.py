import numpy as np
from modelpack.UserData import UserData
from modelpack.ModelData import ModelData
from modelpack.SliceData import SliceData
from modelpack.modelOptimization import optimize
# --------------------
# EXPERIMENT CONSTANTS
# --------------------

# TODO: Get these constants by another hist_aux for the basestation
'''
# Reading the trial scenario (basestation) data
bs_hist = np.load(bs_file_loc_base.format(trial_num=TRIAL))
'''

R = 17 # Available RBGs, ORIGINAL = 17
B = 100.0 * 10**6 # Bandwidth in hertz, ORIGINAL = 100
PS = 1024 # Packet size in bits, ORIGINAL = 8192*8
B_MAX = 1024*64*PS # User buffer capacity in bits, ORIGINAL = 1024
L_MAX = 100 # Maximum packets latency in TTIs (ms) 
WINDOW = 10 # Window size for historical metrics
e = 10**(-5) # Small constant for approximation
use_heavy_requirements = False # Choose heavy or moderate requirements
allocate_all_resources = False # Use restriction to allocate all resources on leave it to be minimized

# Moderate requirements
mod_req = dict()
mod_req["embb"] = dict()
mod_req["urllc"] = dict()
mod_req["be"] = dict()
mod_req["embb"]['r'] = 10 * 10**3 # 10 Megabits/s = 10 Kilobits in this TTI
mod_req["embb"]['l'] = 20.0 # 20ms = 20 TTIs
mod_req["embb"]['p'] = 0.2
mod_req["urllc"]['r'] = 1 * 10**3
mod_req["urllc"]['l'] = 1.0
mod_req["urllc"]['p'] = 1.0 * 10.0**(-5)
mod_req["be"]['g'] = 5 * 10**3
mod_req["be"]['f'] = 2 * 10**3

# Heavy requirements
hvy_req = dict()
hvy_req["embb"] = dict()
hvy_req["urllc"] = dict()
hvy_req["be"] = dict()
hvy_req["embb"]['r'] = 20 * 10**3
hvy_req["embb"]['l'] = 20.0
hvy_req["embb"]['p'] = 0.2
hvy_req["urllc"]['r'] = 5 * 10**3
hvy_req["urllc"]['l'] = 1.0
hvy_req["urllc"]['p'] = 1.0 * 10.0**(-5)
hvy_req["be"]['g'] = 10 * 10**3
hvy_req["be"]['f'] = 5 * 10**3

# Selecting the requirements for the right traffic pattern
if use_heavy_requirements:
    requirements = hvy_req
else:
    requirements = mod_req

# UEs per slice
EMBB_USERS = 4
URLLC_USERS = 3
BE_USERS = 3

# Slice IDs
EMBB_ID = 1
URLLC_ID = 2
BE_ID = 3

TRIAL = 4 # Number of the trial used for generating data for this experiment

# File location strings for UEs, slices and the trial scenario (basestation)
ue_file_loc_base = "./hist/test/sac/ws_1/partial/trial{trial_num}/ues/aux_ue{ue_id}.npz"
slice_file_loc = "./hist/test/sac/ws_1/partial/trial{trial_num}/slices/slice{slice_id}.npz"
bs_file_loc_base = "./hist/test/sac/ws_1/partial/trial{trial_num}/bs.npz"


# ------------------------
# READING TRIAL DATA FILES
# ------------------------
data = ModelData(
    B=B,
    R=R,
    PS=PS,
    e=e,
    l_max=L_MAX,
    w_max=WINDOW
)

# Reading UE files
ue_hist_per_slice = {"embb":[], "urllc":[], "be":[]}
for u in range(1,EMBB_USERS + URLLC_USERS + BE_USERS + 1):
    ue_hist = np.load(ue_file_loc_base.format(trial_num=TRIAL, ue_id=u))
    ue_hist_per_slice[str(ue_hist["slice"])].append(ue_hist)

STEPS = len(ue_hist_per_slice["embb"][0]["real_served_thr"])

# Adding slice and UE data
for s in ue_hist_per_slice.keys():
    hist_r_slice = np.zeros(STEPS)
    hist_d_slice = np.zeros(STEPS)
    hist_rcv_slice = np.zeros(STEPS)
    hist_part_slice = np.zeros(STEPS)
    hist_buff_slice = np.vstack([[np.zeros(L_MAX+1)]*STEPS])
    hist_sent_slice = np.vstack([[np.zeros(L_MAX+1)]*STEPS])
    
    for ue_hist in ue_hist_per_slice[s]:
        hist_r_slice += ue_hist["real_served_thr"]
        hist_d_slice += ue_hist["dropp_pkts"]
        hist_rcv_slice += ue_hist["rcv_pkts"]
        hist_part_slice += ue_hist["part_pkts"]
        hist_buff_slice += ue_hist["buff_pkts"]
        hist_sent_slice += ue_hist["sent_pkts"]

    slice = SliceData(
        id=s,
        b_s_max=B_MAX*len(ue_hist_per_slice[s]),
        hist_r=hist_r_slice,
        hist_d=hist_d_slice,
        hist_rcv=hist_rcv_slice,
        hist_buff=hist_buff_slice,
        hist_sent=hist_sent_slice,
        hist_part=hist_part_slice
    )

    if s == "embb":
        slice.r_req=requirements["embb"]['r']
        slice.l_req=requirements["embb"]['l']
        slice.p_req=requirements["embb"]['p']
    elif s == "urllc":
        slice.r_req=requirements["urllc"]['r']
        slice.l_req=requirements["urllc"]['l']
        slice.p_req=requirements["urllc"]['p']
    elif s == "be":
        slice.f_req=requirements["be"]['f']
        slice.g_req=requirements["be"]['g']

    for ue_hist in ue_hist_per_slice[s]:
        u = UserData(
            id=int(ue_hist["id"]),
            s=s,
            SE=ue_hist["se"]
        )
        slice.addUser(u)

    data.addSlice(slice) 

# -----------------
# SOLVING THE MODEL
# -----------------
feasible=[0]*STEPS
feasible_solutions = 0
unfeasible_solutions = 0
flag_first_unfeasible = True
first_unfeasible = -1
for i in range(STEPS):
    print("STEP",i)
    m, results = optimize(data=data, method="cplex", allocate_all_resources=allocate_all_resources,verbose=False)
    data.advanceStep()
    if results.solver.termination_condition == "optimal":
        feasible[i]=True
        feasible_solutions+=1
    else:
        feasible[i]=False
        unfeasible_solutions+=1
        if flag_first_unfeasible:
            first_unfeasible = i
            flag_first_unfeasible=False

print(feasible)
print("Feasible solutions=",feasible_solutions)
print("Unfeasible solutions=",unfeasible_solutions)
if not flag_first_unfeasible:
    print("First unfeasible solution at step",first_unfeasible)