import pulp
import random
import time
from ortools.sat.python import cp_model

def build_master_lp(known_schedules, mandatory_surgeries, optional_surgeries,
                    all_surgeons, all_days, K_d, A_ld, SIMPLIFIED_TIMES):
    """
    Builds the Master Problem LP model from the paper.
    """
    prob = pulp.LpProblem("Master_Problem_LP", pulp.LpMaximize)
    
    # Add a check for empty schedules
    if not known_schedules:
        # This can happen if there are no mandatory surgeries. 
        # Return a dummy problem.
        prob += 0, "Empty_Problem"
        return prob

    sched_vars = pulp.LpVariable.dicts(
        "Schedule",
        [s["id"] for s in known_schedules],
        lowBound=0,
        cat='Continuous'
    )
    
    # (1) Objective function
    prob += pulp.lpSum(
        s["B_j"] * sched_vars[s["id"]] for s in known_schedules
    ), "Total_Surgery_Time"

    # (2) Mandatory Surgeries 
    for i in mandatory_surgeries:
        prob += pulp.lpSum(
            sched_vars[s["id"]] for s in known_schedules if i in s["surgeries"]
        ) == 1, f"Pi_i_Mandatory_{i}"

    # (3) Optional Surgeries 
    for i in optional_surgeries:
        prob += pulp.lpSum(
            sched_vars[s["id"]] for s in known_schedules if i in s["surgeries"]
        ) <= 1, f"Pi_i_Optional_{i}"

    # (4) Daily OR Limit 
    for d in all_days:
        prob += pulp.lpSum(
            sched_vars[s["id"]] for s in known_schedules if s["day"] == d
        ) <= K_d[d], f"Pi_d_OR_Limit_{d}"

    # (5) Surgeon Max Hours 
    for l in all_surgeons:
        for d in all_days:
            prob += pulp.lpSum(
                s["surgeon_work"].get(l, 0) * sched_vars[s["id"]]
                for s in known_schedules if s["day"] == d
            ) <= A_ld[(l, d)], f"Pi_ld_Surgeon_Hours_{l}_{d}"
            
    # (6) Surgeon Overlap (Coloring Constraint) 
    for l in all_surgeons:
        for d in all_days:
            for t in SIMPLIFIED_TIMES:
                prob += pulp.lpSum(
                    s["surgeon_busy_times"].get((l, d, t), 0) * sched_vars[s["id"]]
                    for s in known_schedules if s["day"] == d
                ) <= 1, f"Pi_ldt_Surgeon_Overlap_{l}_{d}_{t}"
                
    return prob

def get_initial_schedules(all_surgeries_data, mandatory_surgeries, all_days, DAY_MAP, 
                          SIMPLIFIED_TIMES, K_d, A_ld, OBLIGATORY_CLEANING_TIME, ALL_TIMES):
    """
    --- MODIFIED ---
    Creates one simple schedule FOR EVERY mandatory surgery to ensure
    the initial Master LP is feasible.
    
    This heuristic now greedily packs schedules to be *mutually* feasible.
    """
    initial_schedules = []
    
    if not mandatory_surgeries:
        return [] 
    
    day_duration = ALL_TIMES[-1] + 1
    
    # --- Track resource usage greedily ---
    # Track next available start time for a (surgeon, day)
    surgeon_day_availability = {(s, d): 0 for s,d in A_ld.keys()}
    # Track number of ORs used on each day
    day_or_usage = {d: 0 for d in all_days}

    for surg_id in mandatory_surgeries:
        if surg_id not in all_surgeries_data:
            print(f"Warning: Mandatory surgery {surg_id} not in all_surgeries_data. Skipping.")
            continue
            
        surg_data = all_surgeries_data[surg_id]
        surg_name = surg_data["surgeon"]
        duration = surg_data["duration"]
        surg_deadline_day = surg_data["deadline"]
        
        found_slot = False
        
        # Iterate through all possible days, starting from Day 1
        for day_name in all_days:
            day_num = DAY_MAP[day_name]
            
            # 1. Check if day is valid for deadline
            if day_num > surg_deadline_day:
                continue # Can't do it on this day
                
            # 2. Check if an OR is available
            if day_or_usage[day_name] >= K_d[day_name]:
                continue # No ORs left on this day, try next day

            # 3. Check when this surgeon is free on this day
            start_time = surgeon_day_availability.get((surg_name, day_name), 0)
            finish_time = start_time + duration
            
            # 4. Check if this schedule fits within the surgeon's max hours AND day duration
            if finish_time > A_ld[(surg_name, day_name)] or finish_time > day_duration:
                continue # Surgeon is fully booked this day, try next day
                
            # --- Slot is valid! ---
            found_slot = True
            
            # Update resources
            surgeon_day_availability[(surg_name, day_name)] = finish_time + OBLIGATORY_CLEANING_TIME
            day_or_usage[day_name] += 1

            # Calculate busy times based on the *actual* start_time
            busy_times = {}
            for t_busy in SIMPLIFIED_TIMES:
                if start_time <= t_busy < finish_time:
                    busy_times[(surg_name, day_name, t_busy)] = 1
            
            schedule = {
                "id": f"Initial_Sched_{surg_id}_{day_name}", 
                "B_j": duration, 
                "day": day_name, 
                "surgeries": [surg_id],
                "surgeon_work": {surg_name: duration},
                "surgeon_busy_times": busy_times
            }
            initial_schedules.append(schedule)
            
            # Break from the *inner* loop (day_name)
            break 
            
        if not found_slot:
            # This is bad. The problem is likely infeasible *from the generator*.
            # This can happen if, e.g., a surgeon has 10 mandatory surgeries
            # all due Day 1, but only 8 hours.
            print(f"WARNING: Heuristic could not find a valid slot for mandatory surgery {surg_id}.")
            # We must create a "dummy" schedule anyway to have the column.
            # This will *intentionally* be infeasible, and the LP will fail.
            # This indicates an infeasible *test case*.
            day_name = all_days[0] # Just dump it on day 1
            start_time = 0
            busy_times = {}
            for t_busy in SIMPLIFIED_TIMES:
                if start_time <= t_busy < start_time + duration:
                    busy_times[(surg_name, day_name, t_busy)] = 1
            
            schedule = {
                "id": f"Initial_Sched_{surg_id}_{day_name}_DUMMY", 
                "B_j": duration, 
                "day": day_name, 
                "surgeries": [surg_id],
                "surgeon_work": {surg_name: duration},
                "surgeon_busy_times": busy_times
            }
            initial_schedules.append(schedule)

    return initial_schedules

def extract_dual_prices(prob):
    """
    Extracts all dual prices (pi) from the solved master LP.
    """
    dual_prices = {}
    for name, c in prob.constraints.items():
        dual_prices[name] = c.pi
    return dual_prices

def solve_subproblem(
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
    """
    # print(f"  ...Solving REAL Subproblem for {day}...") # (Optional: uncomment for verbose logging)
    
    SCALING_FACTOR = 1000
    day_num = DAY_MAP[day]
    day_duration = all_times[-1] + 1
    
    valid_surgeries = {
        i: data for i, data in all_surgeries_data.items() 
        if data['deadline'] >= day_num
    }
    valid_surgery_ids = [0] + list(valid_surgeries.keys())
    
    if not valid_surgeries:
        # print(f"  ...No valid surgeries for {day}.")
        return None, 0.0

    max_id = max(all_surgeries_data.keys())
    
    # Create lookup tables for CP-SAT AddElement
    surgeon_map = {s: i+1 for i, s in enumerate(all_surgeons)}
    
    duration_list = [0] * (max_id + 1)
    surgeon_list_int = [0] * (max_id + 1)
    infection_type_list = [0] * (max_id + 1)
    G_di_list = [0] * (max_id + 1)
    
    for i, surg in all_surgeries_data.items():
        duration_list[i] = surg['duration']
        surgeon_list_int[i] = surgeon_map[surg['surgeon']]
        infection_type_list[i] = surg['infection_type']

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
            if i == 0 or j == 0:
                continue
            
            inf_i = infection_type_list[i]
            inf_j = infection_type_list[j]
            
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
        
    for i in valid_surgery_ids:
        if i == 0: continue
        bool_vars = []
        for p in range(num_positions):
            b = model.NewBoolVar(f'b_{i}_{p}')
            model.Add(W[p] == i).OnlyEnforceIf(b)
            model.Add(W[p] != i).OnlyEnforceIf(b.Not())
            bool_vars.append(b)
        model.AddAtMostOne(bool_vars)
        
    for p in range(num_positions - 1):
        is_W_p_zero = model.NewBoolVar(f'is_W_{p}_zero')
        model.Add(W[p] == 0).OnlyEnforceIf(is_W_p_zero)
        model.Add(W[p] != 0).OnlyEnforceIf(is_W_p_zero.Not())
        model.Add(W[p+1] == 0).OnlyEnforceIf(is_W_p_zero)

    for p in range(num_positions - 1):
        cl_p = model.NewIntVar(0, OBLIGATORY_CLEANING_TIME, f'CL_{p}')
        idx_var = model.NewIntVar(0, cl_list_size, f'cl_idx_{p}')
        model.Add(idx_var == W[p] * (max_id + 1) + W[p+1])
        model.AddElement(idx_var, CL_FLAT_LIST, cl_p)
        model.Add(V[p+1] >= V[p] + t_p_vars[p] + cl_p)

    for l_name, l_int in surgeon_map.items():
        t_p_l_vars = [model.NewIntVar(0, day_duration, f't_p_{p}_{l_name}') for p in range(num_positions)]
        for p in range(num_positions):
            b_is_surgeon_l = model.NewBoolVar(f'b_{p}_{l_name}')
            model.Add(s_p_vars[p] == l_int).OnlyEnforceIf(b_is_surgeon_l)
            model.Add(s_p_vars[p] != l_int).OnlyEnforceIf(b_is_surgeon_l.Not())
            model.Add(t_p_l_vars[p] == t_p_vars[p]).OnlyEnforceIf(b_is_surgeon_l)
            model.Add(t_p_l_vars[p] == 0).OnlyEnforceIf(b_is_surgeon_l.Not())
        model.Add(sum(t_p_l_vars) <= A_ld[(l_name, day)])
        
    for p in range(num_positions):
        model.Add(V[p] + t_p_vars[p] <= day_duration)

    G_p_vars = [model.NewIntVar(-10000 * SCALING_FACTOR, 10000 * SCALING_FACTOR, f'G_p_{p}') for p in range(num_positions)]
    pi_star_p_vars = [model.NewIntVar(-10000 * SCALING_FACTOR, 10000 * SCALING_FACTOR, f'pi_star_p_{p}') for p in range(num_positions)]
    
    for p in range(num_positions):
        model.AddElement(W[p], G_di_list, G_p_vars[p])
        pi_star_idx = model.NewIntVar(0, pi_star_list_size, f'pi_idx_{p}')
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
            new_schedule = {
                "id": f"Generated_Sched_{random.randint(100,999)}_{day}",
                "day": day, "surgeries": [], "surgeon_work": {},
                "surgeon_busy_times": {}, "B_j": 0
            }
            
            for p in range(num_positions):
                surgery_id = solver.Value(W[p])
                if surgery_id > 0:
                    surg_data = all_surgeries_data[surgery_id]
                    start_time = solver.Value(V[p])
                    
                    new_schedule["surgeries"].append(surgery_id)
                    new_schedule["B_j"] += surg_data["duration"]
                    surg_name = surg_data["surgeon"]
                    
                    new_schedule["surgeon_work"][surg_name] = \
                        new_schedule["surgeon_work"].get(surg_name, 0) + surg_data["duration"]
                    
                    for t_busy in SIMPLIFIED_TIMES:
                         if start_time <= t_busy < start_time + surg_data["duration"]:
                                new_schedule["surgeon_busy_times"][(surg_name, day, t_busy)] = 1
                                
            # print(f"  ...FOUND new schedule {new_schedule['id']} with RC = {reduced_cost:.4f}")
            return new_schedule, reduced_cost

    # print(f"  ...No profitable new schedule found for {day}.")
    return None, 0.0

def run_optimization_scenario(
    scenario_name,
    all_surgeries_data,
    mandatory_surgeries,
    optional_surgeries,
    all_surgeons,
    all_days,
    DAY_MAP,
    K_d,
    A_ld,
    ALL_TIMES,
    OBLIGATORY_CLEANING_TIME,
    SIMPLIFIED_TIMES
):
    """
    --- MODIFIED ---
    Runs the full Column Generation and final Integer Solve for a given scenario.
    Now returns a consistent dictionary, even on failure.
    """
    print(f"\n--- RUNNING SCENARIO: {scenario_name} ---")
    start_time = time.time()
    
    # --- Create a default results dictionary ---
    # This fixes the KeyError, as all keys will always exist.
    results = {
        "scenario_name": scenario_name,
        "status": "Failed",
        "total_scheduled_time": 0,
        "selected_schedules": [],
        "total_iterations": 0,
        "total_columns_generated": 0,
        "runtime_sec": 0
    }

    known_schedules = get_initial_schedules(
        all_surgeries_data, mandatory_surgeries, all_days, DAY_MAP, 
        SIMPLIFIED_TIMES, K_d, A_ld, OBLIGATORY_CLEANING_TIME, ALL_TIMES
    )
    
    # Store initial number of columns
    results["total_columns_generated"] = len(known_schedules)
    
    iteration = 0
    max_iterations = 15 # Add a safety break for large problems
    
    while True:
        iteration += 1
        print(f"  Iteration {iteration}...")
        
        if iteration > max_iterations:
            print("  Reached max iterations. Stopping CG.")
            break
        
        master_lp = build_master_lp(known_schedules, mandatory_surgeries, optional_surgeries,
                                    all_surgeons, all_days, K_d, A_ld, SIMPLIFIED_TIMES)
        master_lp.solve(pulp.PULP_CBC_CMD(msg=0))
        
        if master_lp.status != pulp.LpStatusOptimal:
            print(f"  Master LP failed to solve (Status: {pulp.LpStatus[master_lp.status]}). Stopping.")
            results["status"] = f"Master LP Error ({pulp.LpStatus[master_lp.status]})"
            results["runtime_sec"] = time.time() - start_time
            results["total_iterations"] = iteration
            return results # Return the partially filled (but valid) results dict
            
        dual_prices = extract_dual_prices(master_lp)
        
        new_schedules_found = False
        for day in all_days:
            new_schedule, reduced_cost = solve_subproblem(
                day, dual_prices, all_surgeries_data, 
                all_surgeons, ALL_TIMES, A_ld, DAY_MAP,
                OBLIGATORY_CLEANING_TIME, SIMPLIFIED_TIMES
            )
            
            if reduced_cost > 1e-6:
                new_schedules_found = True
                if not any(s['id'] == new_schedule['id'] for s in known_schedules):
                    known_schedules.append(new_schedule)
                        
        if not new_schedules_found:
            print(f"  CONVERGENCE REACHED in {iteration} iterations.")
            break
    
    results["total_iterations"] = iteration
    results["total_columns_generated"] = len(known_schedules)

    # --- FINAL INTEGER SOLVE ---
    print("  Solving Final Integer Problem...")
    final_prob = build_master_lp(known_schedules, mandatory_surgeries, optional_surgeries,
                                all_surgeons, all_days, K_d, A_ld, SIMPLIFIED_TIMES)
    
    for v in final_prob.variables():
        v.cat = 'Integer'
        v.upBound = 1
        v.lowBound = 0

    final_prob.solve(pulp.PULP_CBC_CMD(msg=0))

    end_time = time.time()
    
    # --- COLLATE FINAL RESULTS ---
    results["status"] = pulp.LpStatus[final_prob.status]
    results["runtime_sec"] = end_time - start_time

    if final_prob.status == pulp.LpStatusOptimal:
        results["total_scheduled_time"] = pulp.value(final_prob.objective)
        for v in final_prob.variables():
            if v.value() > 0.9:
                for s in known_schedules:
                    if s["id"] == v.name.replace("Schedule_", ""):
                        results["selected_schedules"].append(s)
    
    print(f"--- {scenario_name} FINISHED ({(end_time - start_time):.2f}s) ---")
    return results