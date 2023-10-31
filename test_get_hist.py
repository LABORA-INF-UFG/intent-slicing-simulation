import numpy as np
from modelpack.UserData import UserData
# --------------------
# EXPERIMENT CONSTANTS
# --------------------

# TODO: Get these constants by another hist_aux for the basestation

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

for u in range(1,11):
    ue_hist = np.load(ue_file_loc_base.format(trial_num=TRIAL, ue_id=u))
    print(ue_hist["slice"])

'''
# Reading the trial scenario (basestation) data
bs_hist = np.load(bs_file_loc_base.format(trial_num=TRIAL))

# --------------------------- Reading EMBB data

# Reading the EMBB slice data
embb_hist = np.load(slice_file_loc.format(trial_num=TRIAL, slice_id=EMBB_ID))

# Reading data from EMBB UEs
user_data = list()
for u_id in range(1, EMBB_USERS + 1):
    ue_hist = np.load(ue_file_loc_base.format(trial_num=TRIAL, ue_id=u_id))
    user_data.append(UserData(id=u_id, s="embb", SE=list(ue_hist["se"])))
    if u_id == 1:
        print(ue_hist["pkt_loss"])

    # Generating slice data from its users
    


# --------------------------- Reading URLLC data

# Reading the URLLC slice data
urllc_hist = np.load(slice_file_loc.format(trial_num=TRIAL, slice_id=URLLC_ID))

# Reading data from URLLC UEs
user_data = list()
for u_id in range(EMBB_USERS + 1, EMBB_USERS + URLLC_USERS + 1):
    ue_hist = np.load(ue_file_loc_base.format(trial_num=TRIAL, ue_id=u_id))
    user_data.append(UserData(id=u_id, s="embb", SE=list(ue_hist["se"])))

# --------------------------- Reading BE data

# Reading the BE slice data
be_hist = np.load(slice_file_loc.format(trial_num=TRIAL, slice_id=BE_ID))

# Reading data from BE UEs
user_data = list()
for u_id in range(EMBB_USERS + URLLC_USERS + 1, EMBB_USERS + URLLC_USERS + BE_USERS + 1):
    ue_hist = np.load(ue_file_loc_base.format(trial_num=TRIAL, ue_id=u_id))
    user_data.append(UserData(id=u_id, s="embb", SE=list(ue_hist["se"])))


'''







# HIST DATA NOW
# - BASESTATION
# - - actions
# - - rewards
# - SLICES (mean of users)
# - - pkt_rcv
# - - pkt_snt
# - - pkt_thr
# - - buffer_occ
# - - avg_lat
# - - pkt_loss
# - - se
# - - long_term_pkt_thr
# - - fifth_perc_pkt_thr
# - USERS
# - - pkt_rcv
# - - pkt_snt
# - - pkt_thr
# - - buffer_occ
# - - avg_lat
# - - pkt_loss
# - - se
# - - long_term_pkt_thr
# - - fifth_perc_pkt_thr

# NEEDED DATA FOR EACH STEP
# - BASESTATION
# - - Bandwidth
# - - Total number of RBGs
# - - Packet size
# - - Window size for aggregated metrics
# - - l_max
# - - b_max
# - - Name and id of each slice
# - Slice
# - - Metric requirements (have to implement)
# - - Id of all users in the slice
# - - Historical slice served throughput (sum all users throughputs)
# - - Buffer packets (sum of user buffers)
# - - Cumulative sent packets (sum of user cumulative sent packets)
# - - Request: packets received by the buffer, including dropped ones (sum of users requests)
# - - Number of packets in the buffer (sum the buffer)
# - - Dropped packets (sum of user dropped packets)
# - Users
# - - Spectral efficiency
# - - Served throughput (pkt_thr + pkt_part)
# - - pkt_thr
# - - pkt_part (have to implement)
# - - Buffer packets right after receiving new packets (have to implement)
# - - Cumulative number of sent packets (have to implement)
# - - Requests (have to implement)
# - - Dropped packets (maybe have to implement)