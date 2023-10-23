from modelpack.ModelData import ModelData
from modelpack.SliceData import SliceData
from modelpack.UserData import UserData
from modelpack.modelOptimization import optimize

from timeit import default_timer as timer
from pyomo import environ as pyo


def main():
    # ---------------- Setting up the experiment configuration ----------------

    R = 17 # Available RBGs, ORIGINAL = 17
    B = 100.0 * 10**6 # Bandwidth in hertz, ORIGINAL = 100
    PS = 1024 # Packet size in bits, ORIGINAL = 8192*8
    B_MAX = 1024*64*PS # User buffer capacity in bits, ORIGINAL = 1024
    L_MAX = 100 # Maximum packets latency in TTIs (ms) 
    WINDOW = 10 # Window size for historical metrics
    e = 10**(-5) # Small constant for approximation
    use_heavy_traffic = False # Choose heavy or moderate traffic
    allocate_all_resources = False # Use restriction to allocate all resources on leave it to be minimized

    # UEs per slice
    EMBB_USERS = 4 # ORIGINAL = 4
    URLLC_USERS = 3 # ORIGINAL = 3
    BE_USERS = 3 # ORIGINAL = 3

    # Sets
    slices = ["embb", "urllc", "be"]
    users_ids = range(EMBB_USERS + URLLC_USERS + BE_USERS)

    # Required metrics for moderate traffic
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

    # Required metrics for heavy traffic
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
    if use_heavy_traffic:
        requirements = hvy_req
    else:
        requirements = mod_req

    # REMOVE THIS AFTER TESTING
    #requirements = light_req

    # Creating the eMBB slice
    eMBB = SliceData(
        id="embb",
        b_max=B_MAX*EMBB_USERS,
        l_max=L_MAX,
        r_req=requirements["embb"]['r'],
        l_req=requirements["embb"]['l'],
        p_req=requirements["embb"]['p']
    )
    
    # Creating the URLLC slice
    URLLC = SliceData(
        id="urllc",
        b_max=B_MAX*URLLC_USERS,
        l_max=L_MAX,
        r_req=requirements["urllc"]['r'],
        l_req=requirements["urllc"]['l'],
        p_req=requirements["urllc"]['p']
    )
    
    # Creating the BE slice
    BE = SliceData(
        id="be",
        b_max=B_MAX*BE_USERS,
        l_max=L_MAX,
        g_req=requirements["be"]['g'],
        f_req=requirements["be"]['f']
    )

    # Standard spectralm efficiency for each user
    SE = 1 * 10**-3 # bits/(TTI * Hz)

    # Adding EMBB users
    for i in range (EMBB_USERS): 
        u = UserData(id = i, s = "embb", SE = SE)
        eMBB.addUser(u)
    
    # Adding URLLC users
    for i in range (URLLC_USERS): 
        u = UserData(id = i+EMBB_USERS, s = "urllc", SE = SE)
        URLLC.addUser(u)
    
    # Adding BE users
    for i in range (BE_USERS): 
        u = UserData(id = i+EMBB_USERS+URLLC_USERS, s = "be", SE = SE)
        BE.addUser(u)

    # Adding arriving packets to all slices
    # Each user requests 2 Mb/s = 2kb/TTI 
    eMBB.addArrivingPackets(int(requirements["embb"]['r']/PS) * EMBB_USERS)
    URLLC.addArrivingPackets(int(requirements["urllc"]['r']/PS) * URLLC_USERS)
    # BE doesn't have buffer metrics

    # Formatting the model input data as an object
    modelData = ModelData(B=B, R=R, PS=PS, e=e, l_max=L_MAX, b_max=B_MAX, w_max=WINDOW)

    # Adding slices to the model
    modelData.addSlice(eMBB)
    modelData.addSlice(URLLC)
    modelData.addSlice(BE)

    # ---------------- Bulding and Solving the Model ----------------
    model = optimize(modelData, "cplex", allocate_all_resources=allocate_all_resources)

    # ---------------- Extracting the Results ----------------
    R_s = dict()
    for s in model.S:
        R_s[s] = model.R_s[s].value
    print("RBGs per slice:",R_s)

    R_u = dict()
    for u in model.U:
        R_u[u] = model.R_u[u].value
    print("RBGs per user:",R_u)

import sys
if __name__ == "__main__":
    main()