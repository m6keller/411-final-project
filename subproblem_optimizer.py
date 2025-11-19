from ortools.sat.python import cp_model
from schedule import Schedule
import random

def generate_daily_schedule(
    day, 
    dual_prices, 
    all_surgeries_data, 
    all_surgeons,
    all_times, # This is our day duration
    A_ld,
    DAY_MAP,
    OBLIGATORY_CLEANING_TIME,
    SIMPLIFIED_TIMES
):
    """
    Solves the subproblem for a single 'day' using a CP-SAT model.
    This is the pricing problem in the column generation scheme.
    It generates a new schedule (a column) that has the potential
    to improve the master problem's objective function.
    """
    SCALING_FACTOR = 1000
    day_num = DAY_MAP[day]
    day_duration = all_times[-1] + 1
    
    valid_surgeries = {
        i: data for i, data in all_surgeries_data.items() 
        if data['deadline'] >= day_num
    }
    valid_surgery_ids = [0] + list(valid_surgeries.keys())
    
    if not valid_surgeries:
        return None, 0.0

    max_id = max(all_surgeries_data.keys()) if all_surgeries_data else 0
    
    # Create lookup tables for CP-SAT AddElement
    surgeon_map = {s: i+1 for i, s in enumerate(all_surgeons)}
    
    duration_list = [0] * (max_id + 1)
    surgeon_list_int = [0] * (max_id + 1)
    G_di_list = [0] * (max_id + 1)
    
    for i, surg in all_surgeries_data.items():
        duration_list[i] = surg['duration']
        surgeon_list_int[i] = surgeon_map[surg['surgeon']]

    for i in valid_surgery_ids:
        if i == 0: continue
        surg = valid_surgeries[i]
        pi_i = (
            dual_prices.get(f"Pi_i_Mandatory_{i}", 0) + 
            dual_prices.get(f"Pi_i_Optional_{i}", 0)
        )
        pi_ld = dual_prices.get(f"Pi_ld_Surgeon_Hours_{surg['surgeon']}_{day}", 0)
        t_i = surg['duration']
        
        G_di = t_i - pi_i - (t_i * pi_ld)
        G_di_list[i] = int(G_di * SCALING_FACTOR)
        
    cl_list_size = (max_id + 1) * (max_id + 1)
    CL_FLAT_LIST = [0] * cl_list_size
    for i in range(max_id + 1):
        for j in range(max_id + 1):
            if i == 0 or j == 0 or i not in all_surgeries_data or j not in all_surgeries_data:
                continue
            
            inf_i = all_surgeries_data[i]['infection_type']
            inf_j = all_surgeries_data[j]['infection_type']
            
            if (inf_i > 0 and inf_j == 0) or \
               (inf_i > 0 and inf_j > 0 and inf_i != inf_j):
                CL_FLAT_LIST[i * (max_id + 1) + j] = OBLIGATORY_CLEANING_TIME

    pi_star_list_size = (max_id + 1) * (day_duration + 1)
    PI_STAR_FLAT_LIST = [0] * pi_star_list_size
    
    for i in valid_surgery_ids:
        if i == 0: continue
        surg_data = all_surgeries_data[i]
        s_i = surg_data['surgeon']
        t_i = surg_data['duration']
        
        for t in range(day_duration):
            pi_star_cost = 0
            for t_prime in SIMPLIFIED_TIMES:
                if t <= t_prime < t + t_i:
                    pi_star_cost += dual_prices.get(f"Pi_ldt_Surgeon_Overlap_{s_i}_{day}_{t_prime}", 0)
            
            PI_STAR_FLAT_LIST[i * (day_duration + 1) + t] = int(pi_star_cost * SCALING_FACTOR)

    pi_d_cost = dual_prices.get(f"Pi_d_OR_Limit_{day}", 0)

    model = cp_model.CpModel()
    num_positions = 8
    
    W = [model.NewIntVarFromDomain(cp_model.Domain.FromValues(valid_surgery_ids), f'W_{p}') 
         for p in range(num_positions)]
    V = [model.NewIntVar(0, day_duration, f'V_{p}') 
         for p in range(num_positions)]

    t_p_vars = [model.NewIntVar(0, day_duration, f't_p_{p}') for p in range(num_positions)]
    s_p_vars = [model.NewIntVar(0, len(all_surgeons), f's_p_{p}') for p in range(num_positions)]
    
    for p in range(num_positions):
        model.AddElement(W[p], duration_list, t_p_vars[p])
        model.AddElement(W[p], surgeon_list_int, s_p_vars[p])
        
    # Distinctness: Each surgery scheduled at most once
    for i in valid_surgery_ids:
        if i == 0: continue
        bool_vars = []
        for p in range(num_positions):
            b = model.NewBoolVar(f'b_{i}_{p}')
            model.Add(W[p] == i).OnlyEnforceIf(b)
            model.Add(W[p] != i).OnlyEnforceIf(b.Not())
            bool_vars.append(b)
        model.AddAtMostOne(bool_vars)
        
    # Continuity: If W_p is empty, subsequent positions are empty
    for p in range(num_positions - 1):
        is_W_p_zero = model.NewBoolVar(f'is_W_{p}_zero')
        model.Add(W[p] == 0).OnlyEnforceIf(is_W_p_zero)
        model.Add(W[p] != 0).OnlyEnforceIf(is_W_p_zero.Not())
        model.Add(W[p+1] == 0).OnlyEnforceIf(is_W_p_zero)

    # Timing & Cleaning
    for p in range(num_positions - 1):
        cl_p = model.NewIntVar(0, OBLIGATORY_CLEANING_TIME, f'CL_{p}')
        idx_var = model.NewIntVar(0, cl_list_size -1, f'cl_idx_{p}')
        model.Add(idx_var == W[p] * (max_id + 1) + W[p+1])
        model.AddElement(idx_var, CL_FLAT_LIST, cl_p)
        model.Add(V[p+1] >= V[p] + t_p_vars[p] + cl_p)

    # Surgeon Max Time
    for l_name, l_int in surgeon_map.items():
        t_p_l_vars = [model.NewIntVar(0, day_duration, f't_p_{p}_{l_name}') for p in range(num_positions)]
        for p in range(num_positions):
            b_is_surgeon_l = model.NewBoolVar(f'b_{p}_{l_name}')
            model.Add(s_p_vars[p] == l_int).OnlyEnforceIf(b_is_surgeon_l)
            model.Add(s_p_vars[p] != l_int).OnlyEnforceIf(b_is_surgeon_l.Not())
            model.Add(t_p_l_vars[p] == t_p_vars[p]).OnlyEnforceIf(b_is_surgeon_l)
            model.Add(t_p_l_vars[p] == 0).OnlyEnforceIf(b_is_surgeon_l.Not())
        model.Add(sum(t_p_l_vars) <= A_ld[(l_name, day)])
        
    # Symmetry Breaking
    for p in range(1, num_positions):
        is_W_p_zero = model.NewBoolVar(f'is_W_{p}_zero_for_V')
        model.Add(W[p] == 0).OnlyEnforceIf(is_W_p_zero)
        model.Add(W[p] != 0).OnlyEnforceIf(is_W_p_zero.Not())
        model.Add(V[p] == V[p-1] + t_p_vars[p-1]).OnlyEnforceIf(is_W_p_zero)
        
    # Stay within day
    for p in range(num_positions):
        model.Add(V[p] + t_p_vars[p] <= day_duration)

    # --- Objective ---
    G_p_vars = [model.NewIntVar(-10000 * SCALING_FACTOR, 10000 * SCALING_FACTOR, f'G_p_{p}') for p in range(num_positions)]
    pi_star_p_vars = [model.NewIntVar(-10000 * SCALING_FACTOR, 10000 * SCALING_FACTOR, f'pi_star_p_{p}') for p in range(num_positions)]
    
    for p in range(num_positions):
        model.AddElement(W[p], G_di_list, G_p_vars[p])
        pi_star_idx = model.NewIntVar(0, pi_star_list_size -1, f'pi_idx_{p}')
        model.Add(pi_star_idx == W[p] * (day_duration + 1) + V[p])
        model.AddElement(pi_star_idx, PI_STAR_FLAT_LIST, pi_star_p_vars[p])

    model.Maximize(
        sum(G_p_vars) - 
        sum(pi_star_p_vars) - 
        int(pi_d_cost * SCALING_FACTOR)
    )
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        reduced_cost = solver.ObjectiveValue() / SCALING_FACTOR
        
        if reduced_cost > 1e-6:
            surgeries_in_sched = []
            surgeon_work = {}
            surgeon_busy_times = {}
            B_j = 0

            for p in range(num_positions):
                surgery_id = solver.Value(W[p])
                if surgery_id > 0:
                    surg_data = all_surgeries_data[surgery_id]
                    start_time = solver.Value(V[p])
                    
                    surgeries_in_sched.append(surgery_id)
                    B_j += surg_data["duration"]
                    surg_name = surg_data["surgeon"]
                    
                    surgeon_work[surg_name] = \
                        surgeon_work.get(surg_name, 0) + surg_data["duration"]
                    
                    for t_busy in SIMPLIFIED_TIMES:
                         if start_time <= t_busy < start_time + surg_data["duration"]:
                                surgeon_busy_times[(surg_name, day, t_busy)] = 1
            
            new_schedule = Schedule(
                schedule_id=f"Generated_Sched_{random.randint(100,999)}_{day}",
                day=day, 
                surgeries=surgeries_in_sched, 
                surgeon_work=surgeon_work,
                surgeon_busy_times=surgeon_busy_times, 
                B_j=B_j
            )
            return new_schedule, reduced_cost

    return None, 0.0
