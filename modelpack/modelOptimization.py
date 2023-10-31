from pyomo import environ as pyo

from .SliceData import SliceData
from .ModelData import ModelData

def optimize(data: ModelData, method: str, allocate_all_resources = True,verbose=False):
    '''
    Function for building and solving the linear model.

    Parameters
    ----------
    data: ModelData
        Input data for building the model.
    
    method: str
        Lower case string with the solver's name (e.g. cplex or gurobi)  
    
    allocate_all_resources: bool, optional
        Flag that indicates how to constraint the slice RBGs. If true. then sum(R_s) == R.
        Else, sum(R_s) <= R and the allocation is minimized.

    tee: bool, optional
        Flag for verbose solving.
    
    Returns
    -------
    ConcreteModel
        The built and solved model with values accessible by using m.var_name.value attribute. 
    '''
    if verbose:
        print ("Building model...")

    m = pyo.ConcreteModel()

    # ----
    # SETS
    # ----

    # SET: S
    m.S = pyo.Set(initialize = ["embb", "urllc", "be"])

    # SET: S_rlp
    m.S_rlp = pyo.Set(initialize = ["embb", "urllc"])
    
    # SET: S_fg
    m.S_fg = pyo.Set(initialize = ["be"])

    # SET: U
    all_users = []
    for s in m.S:
        all_users.extend(data.slices[s].users.keys())
    m.U = pyo.Set(initialize = all_users)

    # SET: U_rlp
    rlp_users = []
    for s in m.S_rlp:
        rlp_users.extend(data.slices[s].users.keys())
    m.U_rlp = pyo.Set(initialize = rlp_users)

    # SET: U_rlp
    fg_users = []
    for s in m.S_fg:
        fg_users.extend(data.slices[s].users.keys())
    m.U_fg = pyo.Set(initialize = fg_users)

    # SETS: U_{s}
    m.U_EMBB = pyo.Set(initialize = data.slices["embb"].users.keys())
    m.U_URLLC = pyo.Set(initialize = data.slices["urllc"].users.keys())
    m.U_BE = pyo.Set(initialize = data.slices["be"].users.keys())
    U_s = dict()
    U_s["embb"] = m.U_EMBB
    U_s["urllc"] = m.U_URLLC
    U_s["be"] = m.U_BE

    # SET: i = 0, ..., l_{max}
    m.I = pyo.Set(initialize = range(data.l_max + 1))

    # Set: i = 0, ..., l_{max} - 1
    m.I_0_l_max_1 = pyo.Set(initialize = range(data.l_max))

    # Set: i = 1, ..., l_{max}
    m.I_1_l_max = pyo.Set(initialize = range(1, data.l_max+1))

    # ----
    # VARS
    # ----

    # VAR: R_s
    m.R_s = pyo.Var(m.S, domain=pyo.NonNegativeIntegers)

    # VAR: R_u for all slices
    m.R_u = pyo.Var(m.U, domain=pyo.NonNegativeIntegers)

    # VAR: k_u for all slices
    m.k_s = pyo.Var(m.S, domain=pyo.NonNegativeIntegers)
    
    # VAR: sent_s^i for rlp slices
    m.sent_s_i = pyo.Var(m.S_rlp, m.I, domain=pyo.NonNegativeIntegers)
    
    # VAR: T_s for rlp slices
    m.T_s = pyo.Var(m.S_rlp, domain=pyo.NonNegativeIntegers)

    # VAR: MAXbuff_s_i for rlp slices
    m.MAXbuff_s_i = pyo.Var(m.S_rlp, m.I_1_l_max, domain=pyo.NonNegativeIntegers)

    # VAR: MAXover_s for rlp slices
    m.MAXover_s = pyo.Var(m.S_rlp, domain=pyo.NonNegativeIntegers)

    # VAR: alpha_s for rlp slices
    m.alpha_s = pyo.Var(m.S_rlp, domain=pyo.Binary)

    # VAR: delta_s_i for rlp slices
    m.delta_s_i = pyo.Var(m.S_rlp, m.I_1_l_max, domain=pyo.Binary)

    # VAR: beta_s for rlp slices
    m.beta_s = pyo.Var(m.S_rlp, domain=pyo.Binary)

    if data.n > 0:
        # VAR: psi_s for fg slices if n >= 1
        m.psi_s = pyo.Var(m.S_fg, domain=pyo.Binary)

    if data.n > 1:
        # VAR: rho_s for fg slices if n >= 2
        m.rho_s = pyo.Var(m.S_fg, domain=pyo.Binary)

        # VAR: lambda_s for fg slices if n >= 2
        m.lambda_s = pyo.Var(m.S_fg, domain=pyo.Binary)

        # VAR: sigma_s for fg slices if n >= 2
        m.sigma_s = pyo.Var(m.S_fg, domain=pyo.Binary)

        # VAR: omega_s for fg slices if n >= 2
        m.omega_s = pyo.Var(m.S_fg, domain=pyo.Binary)

    # -----------
    # EXPRESSIONS
    # -----------

    # --------------- Expressions for all slices

    r_u = dict()
    r_s = dict()
    for s in m.S:
        for u in U_s[s]:
            # EXP: r_u calculation for all slices
            r_u[u] = data.B * (m.R_u[u]/data.R) * data.slices[s].users[u].SE[data.n] / 1e3
            
        # EXP: r_s calculation for all slices
        r_s[s] = sum(r_u[u] for u in U_s[s])
    
    # EXP: V_r the upper bound for any r_s
    V_r = data.B * max(max(data.slices[s].users[u].SE[data.n] for u in U_s[s]) for s in m.S)

    # EXP: V_T the upper bound for any T_s
    V_T = max(V_r/data.PS + data.slices[s].b_s_max + data.slices[s].hist_buff[data.n][0] for s in m.S)

    # EXP: V_buff the upper bound of any buff_s_i
    V_buff = max(data.slices[s].b_s_max for s in m.S)

    # EXP: V_over the upper bound of b_s
    V_over = V_buff + max(data.slices[s].hist_rcv[data.n] for s in m.S)

    # --------------- Expressions for rlp slices
    
    remain_s_i = dict()
    b_s_sup = dict()
    over_s = dict()
    d_s_sup = dict()
    p_s = dict()
    for s in m.S_rlp:
        # EXP: remain_s^i = buff_s^i - sent_s^i for rlp slices
        for i in m.I:
            remain_s_i[s,i] = data.slices[s].hist_buff[data.n][i] - m.sent_s_i[s,i]
    
        # EXP: b_s_sup for rlp slices
        b_s_sup[s] = data.slices[s].hist_rcv[data.n] + sum(remain_s_i[s,i] for i in m.I)
    
        # EXP: over_s for rlp slices
        over_s[s] = m.MAXover_s[s] - data.slices[s].b_s_max

        # EXP: d_s for rlp slices
        d_s_sup[s] = remain_s_i[s, data.l_max] + over_s[s]

        # EXP: p_s for rlp slices
        p_s[s] = d_s_sup[s] + sum(data.slices[s].hist_d[data.n-data.w+1 : data.n+1]) / (
            data.slices[s].hist_b_s[data.n-data.w+1] + data.slices[s].hist_rcv[data.n] 
            + sum(data.slices[s].hist_rcv[data.n-data.w+2 : data.n+1])
            )
    
    # --------------- Expressions for fg slices

    g_s = dict()
    for s in m.S_fg:
        # EXP: g_u calculation 
        g_s[s] = (sum(data.slices[s].hist_r[data.n-data.w+1 : data.n]) + r_s[s])/data.w

    # ------------------
    # OBJECTIVE FUNCTION
    # ------------------

    # OBJ: min sum R_s
    m.OBJECTIVE = pyo.Objective(expr=sum(m.R_s[s] for s in m.S), sense=pyo.minimize)

    # -----------
    # CONSTRAINTS
    # -----------

    # --------------- Global constraints

    # CONSTR: sum R_s = R
    if allocate_all_resources:
        m.constr_R_s_sum = pyo.Constraint(expr=sum(m.R_s[s] for s in m.S) == data.R)
    else:
        m.constr_R_s_sum = pyo.Constraint(expr=sum(m.R_s[s] for s in m.S) <= data.R)

    # --------------- Constraints for all slices
    
    m.constr_R_u_sum = pyo.ConstraintList()
    m.constr_R_u_1 = pyo.ConstraintList()
    m.constr_R_u_2 = pyo.ConstraintList()
    
    for s in m.S:
        # CONSTR: sum R_u = R_s
        m.constr_R_u_sum.add(
            sum(m.R_u[u] for u in U_s[s]) == m.R_s[s]
        )

        for u in U_s[s]:
            # CONSTR: R_u intra slice modeling 1
            m.constr_R_u_1.add(
                m.R_u[u] - m.R_s[s]/len(U_s[s]) <= 1 - data.e
            )
            
            # CONSTR: R_u intra slice modeling 2
            m.constr_R_u_2.add(
                m.R_s[s]/len(U_s[s]) - m.R_u[u] <= 1 - data.e
            )

    # --------------- Constraints for fg slices
    
    m.constr_g_s_intent = pyo.ConstraintList()
    m.constr_f_s_intent = pyo.ConstraintList()
    m.constr_psi_s_le = pyo.ConstraintList()
    m.constr_psi_s_ge = pyo.ConstraintList()
    m.constr_rho_s_le = pyo.ConstraintList()
    m.constr_rho_s_ge = pyo.ConstraintList()
    m.constr_lambda_s_le = pyo.ConstraintList()
    m.constr_lambda_s_ge = pyo.ConstraintList()
    m.constr_sigma_s_le = pyo.ConstraintList()
    m.constr_sigma_s_ge = pyo.ConstraintList()
    m.constr_omega_s_le = pyo.ConstraintList()
    m.constr_omega_s_ge = pyo.ConstraintList()
    for s in m.S_fg:
        # CONSTR: Long-term Throughput intent
        m.constr_g_s_intent.add(
            g_s[s] >= data.slices[s].g_req * len(U_s[s])
        )
        
        # Fifth-percentile constraints
        if data.n == 0:
            # CONSTR: Fifth-percentile intent for n = 0
            m.constr_f_s_intent.add(
                r_s[s] >= data.slices[s].f_req * len(U_s[s])
            )    
        elif data.n == 1:
            sort_h = data.slices[s].hist_r[0]
            
            # CONSTR: Psi upper bound
            m.constr_psi_s_le.add(
                r_s[s] + V_r * m.psi_s[s] <= V_r + sort_h
            )
            
            # CONSTR: Psi lower bound
            m.constr_psi_s_ge.add(
                r_s[s] + (sort_h + data.e) * m.psi_s[s] >= sort_h + data.e
            )
            
            # CONSTR: Fifth-percentile intent for n = 1
            m.constr_f_s_intent.add(
                r_s[s] >= m.psi_s[s] * data.slices[s].f_req * len(U_s[s])
            )
        
        else:
            sort = data.slices[s].getSortedThroughputWindow(data.w, data.n)
            h = int((data.w)/20)
            sort_h = sort[h]
            sort_h_1 = sort[h+1]

            # CONSTR: Psi upper bound
            m.constr_psi_s_le.add(
                r_s[s] + V_r * m.psi_s[s] <= V_r + sort_h
            )
            
            # CONSTR: Psi lower bound
            m.constr_psi_s_ge.add(
                r_s[s] + (sort_h + data.e) * m.psi_s[s] >= sort_h + data.e
            )

            # CONSTR: Rho upper bound
            m.constr_rho_s_le.add(
                r_s[s] + V_r * m.rho_s[s] <= V_r + sort_h_1
            )
            
            # CONSTR: Rho lower bound
            m.constr_rho_s_ge.add(
                r_s[s] + (sort_h_1 + data.e) * m.rho_s[s] >= sort_h_1 + data.e
            )

            # CONSTR: Lambda upper bound
            m.constr_lambda_s_le.add(
                m.psi_s[s] + m.rho_s[s] - 2*m.lambda_s[s] >= 0
            )
            
            # CONSTR: Lambda lower bound
            m.constr_lambda_s_ge.add(
                m.psi_s[s] + m.rho_s[s] - data.e * m.lambda_s[s] <= 2 - data.e
            )

            # CONSTR: Sigma upper bound
            m.constr_sigma_s_le.add(
                m.psi_s[s] + m.rho_s[s] + 2*m.sigma_s[s] <= 2
            )
            
            # CONSTR: Sigma lower bound
            m.constr_sigma_s_ge.add(
                m.psi_s[s] + m.rho_s[s] + data.e * m.sigma_s[s] >= data.e
            )

            # CONSTR: Omega upper bound
            m.constr_omega_s_le.add(
                m.lambda_s[s] + m.sigma_s[s] + 2*m.omega_s[s] <= 2
            )
            
            # CONSTR: Omega lower bound
            m.constr_omega_s_ge.add(
                m.lambda_s[s] + m.sigma_s[s] + data.e * m.omega_s[s] >= data.e
            )
            
            # CONSTR: Fifth-percentile intent for n > 1
            m.constr_f_s_intent.add(
                r_s[s] >= m.omega_s[s] * data.slices[s].f_req * len(U_s[s])
            )
            
    
    # --------------- Constraints for rlp slices
    m.constr_k_s_floor_upper = pyo.ConstraintList()
    m.constr_k_s_floor_lower = pyo.ConstraintList()
    m.constr_r_s_intent = pyo.ConstraintList()
    m.constr_sent_l_max = pyo.ConstraintList()
    m.constr_sent_T_s = pyo.ConstraintList()
    m.constr_T_s_le_k_s = pyo.ConstraintList()
    m.constr_T_s_le_sum_buff_s = pyo.ConstraintList()
    m.constr_T_s_ge_k_s = pyo.ConstraintList()
    m.constr_T_s_ge_sum_buff_s = pyo.ConstraintList()
    m.constr_sent_le_delta_buff = pyo.ConstraintList()
    m.constr_maxbuff_ge_buff = pyo.ConstraintList()
    m.constr_maxbuff_ge_sent = pyo.ConstraintList()
    m.constr_maxbuff_le_buff = pyo.ConstraintList()
    m.constr_maxbuff_le_sent = pyo.ConstraintList()
    m.constr_l_s_intent = pyo.ConstraintList()
    m.constr_maxover_s_ge_b_s_sup = pyo.ConstraintList()
    m.constr_maxover_s_ge_b_s_max = pyo.ConstraintList()
    m.constr_maxover_s_le_b_s_sup = pyo.ConstraintList()
    m.constr_maxover_s_le_b_s_max = pyo.ConstraintList()
    m.constr_p_s_intent = pyo.ConstraintList()
    for s in m.S_rlp:
        # CONSTR: Throughput intent
        m.constr_r_s_intent.add(
            r_s[s] >= data.slices[s].r_req * len(U_s[s])
        )

        # CONSTR: k_s flooring upper bound
        m.constr_k_s_floor_upper.add(
            m.k_s[s] <= r_s[s]/data.PS + data.slices[s].hist_part[data.n]
        )

        # CONSTR: k_s flooring lower bound
        m.constr_k_s_floor_lower.add(
            m.k_s[s] + 1 >= r_s[s]/data.PS + data.slices[s].hist_part[data.n] + data.e
        )

        # CONSTR: sent_l_max <= buffer_l_max
        m.constr_sent_l_max.add(
            m.sent_s_i[s,data.l_max] <= data.slices[s].hist_buff[data.n][data.l_max]
        )

        # CONSTR: sum sent_s_i  = T_s
        m.constr_sent_T_s.add(
            sum(m.sent_s_i[s,i] for i in m.I) == m.T_s[s]
        )

        # CONSTR: T_s <= k_s
        m.constr_T_s_le_k_s.add(
            m.T_s[s] <= m.k_s[s]
        )

        # CONSTR: T_s <= sum buff_s
        m.constr_T_s_le_sum_buff_s.add(
            m.T_s[s] <= sum(data.slices[s].hist_buff[data.n][i] for i in m.I)
        )

        # CONSTR: T_s >= k_s
        m.constr_T_s_ge_k_s.add(
            m.T_s[s] >= m.k_s[s] - V_T * (1 - m.alpha_s[s])
        )

        # CONSTR: T_s >= sum buff_s
        m.constr_T_s_ge_sum_buff_s.add(
            m.T_s[s] >= sum(data.slices[s].hist_buff[data.n][i] for i in m.I) - V_T * m.alpha_s[s]
        )
        
        for i in m.I_0_l_max_1:
            # CONSTR: sent_s_i <= delta_s * buff_s
            m.constr_sent_le_delta_buff.add(
                m.sent_s_i[s,i] <= m.delta_s_i[s,i+1] * data.slices[s].hist_buff[data.n][i]
            )
        
        for i in m.I_1_l_max:
            # CONSTR: maxbuff_s >= buff_s
            m.constr_maxbuff_ge_buff.add(
                m.MAXbuff_s_i[s,i] >= data.slices[s].hist_buff[data.n][i]
            )
            
            # CONSTR: maxbuff_s >= sent_s
            m.constr_maxbuff_ge_sent.add(
                m.MAXbuff_s_i[s,i] >= m.sent_s_i[s,i]
            )
            
            # CONSTR: maxbuff_s <= buff_s
            m.constr_maxbuff_ge_buff.add(
                m.MAXbuff_s_i[s,i] <= data.slices[s].hist_buff[data.n][i] + V_buff * (1 - m.delta_s_i[s,i])
            )
            
            # CONSTR: maxbuff_s <= sent_s
            m.constr_maxbuff_ge_buff.add(
                m.MAXbuff_s_i[s,i] <= m.sent_s_i[s,i] + V_buff * m.delta_s_i[s,i]
            )
        
        # CONSTR: Average Buffer Latency intent
        m.constr_l_s_intent.add(
            sum((data.slices[s].hist_acc[data.n][i] + m.sent_s_i[s,i])*i for i in m.I)
            <= data.slices[s].l_req * sum((data.slices[s].hist_acc[data.n][i] + m.sent_s_i[s,i]) for i in m.I)
        )
        
        # CONSTR: maxover_s >= b_s_sup
        m.constr_maxover_s_ge_b_s_sup.add(
            m.MAXover_s[s] >= b_s_sup[s]
        )

        # CONSTR: maxover_s >= b_s_max
        m.constr_maxover_s_ge_b_s_max.add(
            m.MAXover_s[s] >= data.slices[s].b_s_max
        )
        
        # CONSTR: maxover_s <= b_s_sup
        m.constr_maxover_s_le_b_s_sup.add(
            m.MAXover_s[s] <= b_s_sup[s] + V_over * (1 - m.beta_s[s])
        )
        
        # CONSTR: maxover_s <= b_s_max
        m.constr_maxover_s_le_b_s_max.add(
            m.MAXover_s[s] <= data.slices[s].b_s_max + V_over * m.beta_s[s]
        )
        
        # CONSTR: Packet Loss Rate intent
        m.constr_p_s_intent.add(
            p_s[s] <= data.slices[s].p_req
        )
        
        
    # ----------
    # DUAL MODEL
    # ----------

    #m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT) # I don't actually know if this works

    # -------
    # SOLVING
    # -------
    if verbose:
        print("Model built!")
        print("Number of constraints =",m.nconstraints())
        print("Number of variables =",m.nvariables())
        print("Starting solving via {}...".format(method))

    opt = pyo.SolverFactory(method)
    results = opt.solve(m, tee=verbose)
    
    if verbose:
        print("Solved!")

    return m, results