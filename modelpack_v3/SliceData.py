from modelpack.UserData import UserData
import numpy as np

class SliceData:
    def __init__(
            self,
            id: str, # Name of the slice
            l_max: int, # Maximum latency to drop a packet (TTIs, steps or ms)
        ):
        self.id = id
        self.l_max = l_max
        
        self.hist_d = np.array([]) # Historical dropped packets (list of packets)
        self.hist_rcv = np.array([]) # Historical received packets (list of packets)
        self.hist_r = np.array([]) # Historical served throughput (list of Mb/s or Kb/ms)
        self.hist_part = np.array([]) # Historical partially sent packets (list of part of packets)
        self.hist_buff = np.ndarray((0,self.l_max+1)) # Historical buffer packets (list of list of packets)
        self.hist_sent = np.ndarray((0,self.l_max+1)) # Historical sent packets (list of list of packets)

        # Initializing a dictionary for saving users
        self.users = dict()

        # Maximum buffer size for the slice buffer, i.e. for all user buffers summed (packets)
        # Calculated as users are added
        self.b_s_max = 0

        # Initializing the number of packets on the buffer at the beggining of each step,
        # considering packets that tried to arrive the buffer but were dropped (list of packets)
        self.hist_b_s = np.array([])
        
        # Initializing the accumulated sent packets of previous steps (list of list of packets)
        self.acc = np.zeros(l_max+1)
        self.hist_acc = np.ndarray((0,l_max+1))

        # The round robin prioritization of ues in the slice
        self.rr_prioritization = []

    # Associates user with the slice
    def addUser (self, u: UserData) -> None:
        self.users[u.id] = u
        self.b_s_max += u.b_max
    
    # Disassociates all users from the slice
    def disassociateUsers(self):
        self.users = dict()

    # Returns a sorted list of throughputs in a window that ends in step - 1
    def getSortedThroughputWindow (self, w, n):
        return sorted(self.hist_r[n-w+1:n])
    
    # Updates the slice data before the simulation step and model soving
    def updateHistBefStep(self, step: int):
        d = 0
        rcv = 0
        part = 0
        b_s = 0
        buff = np.zeros(self.l_max+1)
        
        for u in self.users.values():
            d += u.hist_d[step]
            rcv += u.hist_rcv[step]
            part += u.hist_part[step]
            buff += u.hist_buff[step]
            b_s += u.hist_b[step]

        self.hist_d = np.append(self.hist_d, d)
        self.hist_rcv = np.append(self.hist_rcv, rcv)
        self.hist_part = np.append(self.hist_part, part)
        self.hist_buff = np.vstack([self.hist_buff, buff])
        self.hist_b_s = np.append(self.hist_b_s, rcv - buff[0] + sum(buff))
        self.hist_acc = np.vstack([self.hist_acc, self.acc])    
    
    # Updates the slice data after the simulation step and model soving
    def updateHistAftStep(self, step: int):
        r = 0
        sent = np.zeros(self.l_max+1)
        
        for u in self.users.values():
            r += u.hist_r[step]
            sent += u.hist_sent[step]

        self.hist_r = np.append(self.hist_r, r)
        self.hist_sent=np.vstack([self.hist_sent, sent])
        self.acc += sent