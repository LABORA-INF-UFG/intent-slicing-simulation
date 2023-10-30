import numpy as np

ue1_file_loc = "./hist/test/sac/ws_1/partial/trial4/ues/ue1.npz"
ue1_hist = np.load(ue1_file_loc)

bs_file_loc = "./hist/test/sac/ws_1/partial/trial4/bs.npz"
bs_hist = np.load(bs_file_loc)

slice1_file_loc = "./hist/test/sac/ws_1/partial/trial4/slices/slice1.npz"
slice1_hist = np.load(slice1_file_loc)

for a in slice1_hist:
    print(a)

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