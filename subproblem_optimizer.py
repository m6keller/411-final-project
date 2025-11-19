from ortools.sat.python import cp_model
from schedule import Schedule, Surgery
import random

def generate_daily_schedule(
    day, 
    dual_prices, 
    all_surgeries_data, 
    all_surgeons,
    all_times, 
    A_ld,
    DAY_MAP,
    OBLIGATORY_CLEANING_TIME,
    SIMPLIFIED_TIMES
):
    """
    Solves the pricing subproblem for a single 'day' using CP-SAT.
    Includes logic for Deadlines and Infection-based Cleaning times.
    """
    SCALING_FACTOR = 1000
    day_num = DAY_MAP[day]
    day_duration = all_times[-1] + 1
    
    # 1. Filter Surgeries by Deadline
    valid_surgeries = {}
    for i, data in all_surgeries_data.items():
        surg_obj = data['surgery_object']
        # Only consider surgeries whose deadline is NOT passed
        if surg_obj.deadline >= day_num:
            valid_surgeries[i] = data

    valid_surgery_ids = [0] + list(valid_surgeries.keys())
    
    if not valid_surgeries:
        return None, 0.0

    max_id = max(all_surgeries_data.keys()) if all_surgeries_data else 0
    surgeon_map = {s: i+1 for i, s in enumerate(all_surgeons)}
    
    # 2. Pre-compute Constants
    duration_list = [0] * (max_id + 1)
    surgeon_list_int = [0] * (max_id + 1)
    G_di_list = [0] * (max_id + 1)
    
    for i, data in all_surgeries_data.items():
        surg = data['surgery_object']
        duration_list[i] = surg.duration
        surgeon_list_int[i] = surgeon_map[surg.surgeon]

        # Calculate G_di (Reduced Cost)
        pi_i = (
            dual_prices.get(f"Pi_i_Mandatory_{i}", 0) + 
            dual_prices.get(f"Pi_i_Optional_{i}", 0)
        )
        pi_ld = dual_prices.get(f"Pi_ld_Surgeon_Hours_{surg.surgeon}_{day}", 0)
        
        G_di = surg.duration - pi_i - (surg.duration * pi_ld)
        G_di_list[i] = int(G_di * SCALING_FACTOR)

    # [cite_start]3. Build Cleaning Matrix [cite: 95-96]
    cl_list_size = (max_id + 1) * (max_id + 1)
    CL_FLAT_LIST = [0] * cl_list_size
    
    for i in range(max_id + 1):
        for j in range(max_id + 1):
            if i == 0 or j == 0 or i not in all_surgeries_data or j not in all_surgeries_data:
                continue
            
            inf_i = all_surgeries_data[i]['surgery_object'].infection_type
            inf_j = all_surgeries_data[j]['surgery_object'].infection_type
            
            needs_cleaning = False
            if inf_i > 0 and inf_j == 0:
                needs_cleaning = True
            elif inf_i > 0 and inf_j > 0 and inf_i != inf_j:
                needs_cleaning = True
                
            if needs_cleaning:
                CL_FLAT_LIST[i * (max_id + 1) + j] = OBLIGATORY_CLEANING_TIME

    # 4. Pi Star (Surgeon Overlap Costs)
    pi_star_list_size = (max_id + 1) * (day_duration + 1)
    PI_STAR_FLAT_LIST = [0] * pi_star_list_size
    
    for i in valid_surgery_ids:
        if i == 0: continue
        surg = valid_surgeries[i]['surgery_object']
        s_i = surg.surgeon
        t_i = surg.duration
        
        for t in range(day_duration):
            pi_star_cost = 0
            for t_prime in SIMPLIFIED_TIMES:
                if t <= t_prime < t + t_i:
                    pi_star_cost += dual_prices.get(f"Pi_ldt_Surgeon_Overlap_{s_i}_{day}_{t_prime}", 0)
            
            PI_STAR_FLAT_LIST[i * (day_duration + 1) + t] = int(pi_star_cost * SCALING_FACTOR)

    pi_d_cost = dual_prices.get(f"Pi_d_OR_Limit_{day}", 0)

    # 5. Build CP Model
    model = cp_model.CpModel()
    num_positions = 6
    
    W = [model.NewIntVarFromDomain(cp_model.Domain.FromValues(valid_surgery_ids), f'W_{p}') 
         for p in range(num_positions)]
    V = [model.NewIntVar(0, day_duration, f'V_{p}') 
         for p in range(num_positions)]

    t_p_vars = [model.NewIntVar(0, day_duration, f't_p_{p}') for p in range(num_positions)]
    s_p_vars = [model.NewIntVar(0, len(all_surgeons) + 1, f's_p_{p}') for p in range(num_positions)]
    
    for p in range(num_positions):
        model.AddElement(W[p], duration_list, t_p_vars[p])
        model.AddElement(W[p], surgeon_list_int, s_p_vars[p])
        
    # Constraint: Distinctness
    for i in valid_surgery_ids:
        if i == 0: continue
        vars_equal_i = [model.NewBoolVar(f'w_{p}_is_{i}') for p in range(num_positions)]
        for p in range(num_positions):
            model.Add(W[p] == i).OnlyEnforceIf(vars_equal_i[p])
            model.Add(W[p] != i).OnlyEnforceIf(vars_equal_i[p].Not())
        model.Add(sum(vars_equal_i) <= 1)
        
    # --- FIXED SECTION: Compactness Constraint ---
    # "If W[p] is 0, then W[p+1] must be 0"
    for p in range(num_positions - 1):
        # 1. Create a boolean variable that is TRUE if W[p] == 0
        is_p_zero = model.NewBoolVar(f'is_p_zero_{p}')
        model.Add(W[p] == 0).OnlyEnforceIf(is_p_zero)
        model.Add(W[p] != 0).OnlyEnforceIf(is_p_zero.Not())
        
        # 2. Use that boolean to enforce W[p+1] == 0
        model.Add(W[p+1] == 0).OnlyEnforceIf(is_p_zero)

    # Constraint: Sequence Timing WITH CLEANING
    for p in range(num_positions - 1):
        cl_p = model.NewIntVar(0, OBLIGATORY_CLEANING_TIME, f'CL_{p}')
        
        row_offset = model.NewIntVar(0, max_id * (max_id + 1), f'row_offset_{p}')
        model.AddMultiplicationEquality(row_offset, [W[p], model.NewConstant(max_id + 1)])
        
        flat_idx = model.NewIntVar(0, cl_list_size - 1, f'cl_idx_{p}')
        model.Add(flat_idx == row_offset + W[p+1])
        model.AddElement(flat_idx, CL_FLAT_LIST, cl_p)
        
        # Create explicit bool for "is active" (W[p] != 0)
        is_active = model.NewBoolVar(f'active_{p}')
        model.Add(W[p] != 0).OnlyEnforceIf(is_active)
        model.Add(W[p] == 0).OnlyEnforceIf(is_active.Not())
        
        model.Add(V[p+1] >= V[p] + t_p_vars[p] + cl_p).OnlyEnforceIf(is_active)

    # Constraint: Surgeon Max Time
    for l_name, l_int in surgeon_map.items():
        durations_for_l = []
        for p in range(num_positions):
            is_l = model.NewBoolVar(f'is_surgeon_{l_name}_{p}')
            model.Add(s_p_vars[p] == l_int).OnlyEnforceIf(is_l)
            model.Add(s_p_vars[p] != l_int).OnlyEnforceIf(is_l.Not())
            
            dur_var = model.NewIntVar(0, day_duration, f'dur_{l_name}_{p}')
            model.Add(dur_var == t_p_vars[p]).OnlyEnforceIf(is_l)
            model.Add(dur_var == 0).OnlyEnforceIf(is_l.Not())
            durations_for_l.append(dur_var)
        
        model.Add(sum(durations_for_l) <= A_ld[(l_name, day)])
        
    # Constraint: End of Day
    for p in range(num_positions):
        model.Add(V[p] + t_p_vars[p] <= day_duration)

    # 6. Objective
    obj_terms = []
    for p in range(num_positions):
        g_var = model.NewIntVar(-10000 * SCALING_FACTOR, 10000 * SCALING_FACTOR, f'G_{p}')
        model.AddElement(W[p], G_di_list, g_var)
        obj_terms.append(g_var)

        pi_star_var = model.NewIntVar(0, 10000 * SCALING_FACTOR, f'pi_star_{p}')
        idx_pi = model.NewIntVar(0, pi_star_list_size - 1, f'idx_pi_{p}')
        
        row_offset_pi = model.NewIntVar(0, max_id * (day_duration + 1), f'row_pi_{p}')
        model.AddMultiplicationEquality(row_offset_pi, [W[p], model.NewConstant(day_duration + 1)])
        
        model.Add(idx_pi == row_offset_pi + V[p])
        model.AddElement(idx_pi, PI_STAR_FLAT_LIST, pi_star_var)
        obj_terms.append(-pi_star_var)

    obj_terms.append(-int(pi_d_cost * SCALING_FACTOR))

    model.Maximize(sum(obj_terms))
    
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        reduced_cost = solver.ObjectiveValue() / SCALING_FACTOR
        
        if reduced_cost > 1e-5:
            surgeries_in_sched = []
            surgeries_objects = []
            surgeon_work = {}
            start_times_dict = {}
            surgeon_busy_times = {} 
            B_j = 0

            for p in range(num_positions):
                surgery_id = solver.Value(W[p])
                if surgery_id > 0:
                    surg_obj = all_surgeries_data[surgery_id]['surgery_object']
                    start_time = solver.Value(V[p])
                    duration = surg_obj.duration
                    surg_name = surg_obj.surgeon
                    
                    surgeries_in_sched.append(surgery_id)
                    surgeries_objects.append(surg_obj)
                    start_times_dict[surgery_id] = start_time
                    
                    B_j += duration
                    surgeon_work[surg_name] = surgeon_work.get(surg_name, 0) + duration
                    
                    surgeon_busy_times[(surg_name, start_time)] = duration
            
            new_schedule = Schedule(
                id=f"Gen_{day}_{random.randint(1000,9999)}",
                day=day, 
                surgeries=surgeries_in_sched, 
                surgeries_data=surgeries_objects,
                surgeon_work=surgeon_work,
                surgeon_busy_times=surgeon_busy_times, 
                start_times=start_times_dict,
                B_j=B_j
            )
            return new_schedule, reduced_cost

    return None, 0.0