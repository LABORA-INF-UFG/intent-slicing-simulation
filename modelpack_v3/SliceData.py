from modelpack.UserData import UserData
import numpy as np

class SliceData:
    def __init__(
            self,
            id: str, # Name of the slice
            l_max: int, # Maximum latency to drop a packet (TTIs, steps or ms)
            hist_r: list, # Historical served throughput (list of Mb/s or Kb/ms)
            hist_d: list, # Historical dropped packets (list of packets)
            hist_rcv: list, # Historical received packets (list of packets)
            hist_buff: list, # Historical buffer packets (list of list of packets)
            hist_sent: list, # Historical sent packets (list of list of packets)
            hist_part: list, # Historical partially sent packets (list of part of packets)
            r_req: float = None, # Minimum required served throughput for each user (Mb/s or Kb/ms)
            l_req: float = None, # Maximum required average buffer latency (ms or TTIs or steps)
            p_req: float = None, # Maximum required packet loss rate (ratio)
            f_req: float = None, # Minimum required fifth-percentile served throughput (Mb/s or Kb/s)
            g_req: float = None # Minimum required long-term served throughput (Mb/s or Kb/s)
        ):
        self.id = id
        self.hist_d = hist_d
        self.hist_buff = hist_buff
        self.hist_rcv = hist_rcv
        self.hist_r = hist_r
        self.hist_sent = hist_sent
        self.hist_part = hist_part
        self.r_req = r_req
        self.l_req = l_req
        self.p_req = p_req
        self.f_req = f_req
        self.g_req = g_req
        
        # Maximum buffer size for the slice buffer, i.e. for all user buffers summed (packets)
        # Calculated as users are added
        self.b_s_max = 0

        # Initializing a dictionary for saving users
        self.users = dict()

        # Initializing the number of packets on the buffer at the beggining of each step,
        # considering packets that tried to arrive the buffer but were dropped (list of packets)
        self.hist_b_s = np.array([])
        
        # Initializing the accumulated sent packets of previous steps (list of list of packets)
        self.acc = np.zeros(l_max+1)
        self.hist_acc = np.ndarray((0,l_max+1))

    # Associates user with the slice
    def addUser (self, u: UserData) -> None:
        self.users[u.id] = u
        self.b_s_max += u.b_max
    
    # Returns a sorted list of throughputs in a window that ends in step - 1
    def getSortedThroughputWindow (self, w, n):
        return sorted(self.hist_r[n-w+1:n])
    
    # Updates the slice data before the simulation step and model soving
    def updateHistBefStep(
        self,
        d:int,
        rcv:int,
        part:float,
        buff:np.array,
        ):
        self.hist_d = np.append(self.hist_d, d)
        self.hist_rcv = np.append(self.hist_rcv, rcv)
        self.hist_part = np.append(self.hist_part, part)
        self.hist_buff = np.vstack([self.hist_buff, buff])
        self.hist_b_s = np.append(self.hist_b_s, rcv - buff[0] + sum(buff))
        self.hist_acc = np.vstack([self.hist_acc, self.acc])    
    
    # Updates the slice data after the simulation step and model soving
    def updateHistAftStep(
        self,
        r: int,
        sent:np.array
        ):
        self.hist_r = np.append(self.hist_r, r)
        self.hist_sent=np.vstack([self.hist_sent, sent])
        self.acc += sent