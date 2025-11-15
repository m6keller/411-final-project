import pulp
import random
from ortools.sat.python import cp_model

# --- 1. FAKE DATA (Updated with deadline and infection info) ---

# Infection Type: 0 = Non-infectious, 1 = Type A, 2 = Type B
# Deadline: Day number (1-indexed)
ALL_SURGERIES_DATA = {
    101: {"duration": 120, "surgeon": "Dr_A", "deadline": 3, "infection_type": 0},
    102: {"duration": 120, "surgeon": "Dr_B", "deadline": 3, "infection_type": 1},
    103: {"duration": 300, "surgeon": "Dr_A", "deadline": 1, "infection_type": 1},
    104: {"duration": 60,  "surgeon": "Dr_A", "deadline": 5, "infection_type": 2},
    105: {"duration": 180, "surgeon": "Dr_B", "deadline": 5, "infection_type": 0},
}
# Obligatory Cleaning Time (in minutes) 
OCT = 30


# --- FAKE Problem Parameters ---
MANDATORY_SURGERIES = [101, 102, 103]
OPTIONAL_SURGERIES = [104, 105]
ALL_SURGEONS = ["Dr_A", "Dr_B"]
ALL_DAYS = ["Mon", "Tue"]
DAY_MAP = {"Mon": 1, "Tue": 2} # Map day name to day number for deadlines
# Assuming 8-hour day (480 minutes)
ALL_TIMES = list(range(0, 480)) 
K_d = {"Mon": 5, "Tue": 5}
A_ld = {
    ("Dr_A", "Mon"): 480, ("Dr_B", "Mon"): 480,
    ("Dr_A", "Tue"): 480, ("Dr_B", "Tue"): 480,
}
# Use the same simplified times for the overlap constraint
SIMPLIFIED_TIMES = list(range(0, 480, 30))

# --- 2. MASTER PROBLEM FUNCTION (Unchanged) ---
def build_master_lp(known_schedules, mandatory_surgeries, optional_surgeries,
                    all_surgeons, all_days, all_times, K_d, A_ld):
    """
    Builds the Master Problem LP model from the paper.
    """
    prob = pulp.LpProblem("Master_Problem_LP", pulp.LpMaximize)
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
    # Using the simplified time buckets from your example
    for l in all_surgeons:
        for d in all_days:
            for t in SIMPLIFIED_TIMES:
                prob += pulp.lpSum(
                    s["surgeon_busy_times"].get((l, d, t), 0) * sched_vars[s["id"]]
                    for s in known_schedules if s["day"] == d
                ) <= 1, f"Pi_ldt_Surgeon_Overlap_{l}_{d}_{t}"
                
    return prob

# --- 3. HELPER FUNCTIONS (Unchanged) ---

def get_initial_schedules():
    """
    Creates one or more simple schedules to start the algorithm.
    """
    schedule_1 = {
        "id": "Initial_Sched_1_Mon", "B_j": 120, "day": "Mon", "surgeries": [101],
        "surgeon_work": {"Dr_A": 120},
        "surgeon_busy_times": {("Dr_A", "Mon", 0): 1, ("Dr_A", "Mon", 30): 1, ("Dr_A", "Mon", 60): 1, ("Dr_A", "Mon", 90): 1}
    }
    schedule_2 = {
        "id": "Initial_Sched_2_Mon", "B_j": 300, "day": "Mon", "surgeries": [103],
        "surgeon_work": {"Dr_A": 300},
        "surgeon_busy_times": {("Dr_A", "Mon", 120): 1, ("Dr_A", "Mon", 150): 1, ("Dr_A", "Mon", 180): 1, ("Dr_A", "Mon", 210): 1, ("Dr_A", "Mon", 240): 1, ("Dr_A", "Mon", 270): 1}
    }
    schedule_3 = {
        "id": "Initial_Sched_3_Tue", "B_j": 120, "day": "Tue", "surgeries": [102],
        "surgeon_work": {"Dr_B": 120},
        "surgeon_busy_times": {("Dr_B", "Tue", 0): 1, ("Dr_B", "Tue", 30): 1, ("Dr_B", "Tue", 60): 1, ("Dr_B", "Tue", 90): 1}
    }
    return [schedule_1, schedule_2, schedule_3]

def extract_dual_prices(prob):
    """
    Extracts all dual prices (pi) from the solved master LP.
    """
    dual_prices = {}
    for name, c in prob.constraints.items():
        dual_prices[name] = c.pi
    return dual_prices

# --- 4. REAL SUBPROBLEM IMPLEMENTATION (Updated) ---

def solve_subproblem(
    day, 
    dual_prices, 
    all_surgeries_data, 
    all_surgeons,
    all_times, # This is our day duration, e.g., list(range(0, 480))
    A_ld
):
    """
    Solves the subproblem for a single 'day' using a CP-SAT model.
    This finds the new schedule (column) with the highest positive reduced cost.
    """
    print(f"  ...Solving REAL Subproblem for {day}...")
    
    # --- 1. PREPROCESSING & LOOKUP TABLE CREATION ---
    SCALING_FACTOR = 1000
    day_num = DAY_MAP[day]
    day_duration = all_times[-1] + 1
    
    # Eq (8): Filter surgeries by deadline 
    valid_surgeries = {
        i: data for i, data in all_surgeries_data.items() 
        if data['deadline'] >= day_num
    }
    valid_surgery_ids = [0] + list(valid_surgeries.keys())
    
    # Use max_id from *all* surgeries for stable indexing
    max_id = max(all_surgeries_data.keys())
    
    # Create lookup tables for CP-SAT AddElement
    # (CP-SAT needs 0-indexed lists for lookups)
    surgeon_map = {s: i+1 for i, s in enumerate(all_surgeons)}
    surgeon_map_inv = {i+1: s for i, s in enumerate(all_surgeons)}
    
    duration_list = [0] * (max_id + 1)
    surgeon_list_int = [0] * (max_id + 1)
    infection_type_list = [0] * (max_id + 1)
    G_di_list = [0] * (max_id + 1)
    
    # Populate basic lookup lists
    for i, surg in all_surgeries_data.items():
        duration_list[i] = surg['duration']
        surgeon_list_int[i] = surgeon_map[surg['surgeon']]
        infection_type_list[i] = surg['infection_type']

    # Populate G_di list (Profit) 
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
        
    # Populate CL_FLAT_LIST (Cleaning Time) 
    # Flattened 2D array: index = i * (max_id + 1) + j
    cl_list_size = (max_id + 1) * (max_id + 1)
    CL_FLAT_LIST = [0] * cl_list_size
    for i in range(max_id + 1):
        for j in range(max_id + 1):
            if i == 0 or j == 0: # No cleaning before/after empty slot 
                continue
                
            inf_i = infection_type_list[i]
            inf_j = infection_type_list[j]
            
            # Cleaning required if:
            # 1. Switching from infectious to non-infectious 
            if inf_i > 0 and inf_j == 0:
                CL_FLAT_LIST[i * (max_id + 1) + j] = OCT
            # 2. Switching from infectious to different infection type 
            elif inf_i > 0 and inf_j > 0 and inf_i != inf_j:
                CL_FLAT_LIST[i * (max_id + 1) + j] = OCT

    # Populate PI_STAR_FLAT_LIST (Start Time Cost) 
    # Flattened 2D array: index = i * (day_duration + 1) + t
    pi_star_list_size = (max_id + 1) * (day_duration + 1)
    PI_STAR_FLAT_LIST = [0] * pi_star_list_size
    
    for i in valid_surgery_ids:
        if i == 0: continue
        surg_data = all_surgeries_data[i]
        s_i = surg_data['surgeon']
        t_i = surg_data['duration']
        
        for t in range(day_duration):
            pi_star_cost = 0
            # Sum duals for all simplified time blocks this surgery overlaps
            for t_prime in SIMPLIFIED_TIMES:
                if t <= t_prime < t + t_i:
                    pi_star_cost += dual_prices.get(f"Pi_ldt_Surgeon_Overlap_{s_i}_{day}_{t_prime}", 0)
            
            PI_STAR_FLAT_LIST[i * (day_duration + 1) + t] = int(pi_star_cost * SCALING_FACTOR)

    pi_d_cost = dual_prices.get(f"Pi_d_OR_Limit_{day}", 0)

    # --- 2. MODEL & VARIABLE SETUP ---
    model = cp_model.CpModel()
    num_positions = 8 # Max number of surgeries per OR
    
    # W_p: Surgery ID in position p 
    W = [model.NewIntVarFromDomain(cp_model.Domain.FromValues(valid_surgery_ids), f'W_{p}') 
         for p in range(num_positions)]
    # V_p: Start time of surgery in position p 
    V = [model.NewIntVar(0, day_duration, f'V_{p}') 
         for p in range(num_positions)]

    # Intermediate variables for lookups
    t_p_vars = [model.NewIntVar(0, day_duration, f't_p_{p}') for p in range(num_positions)]
    s_p_vars = [model.NewIntVar(0, len(all_surgeons), f's_p_{p}') for p in range(num_positions)]
    
    for p in range(num_positions):
        model.AddElement(W[p], duration_list, t_p_vars[p])
        model.AddElement(W[p], surgeon_list_int, s_p_vars[p])
        
    # --- 3. ADD CONSTRAINTS (Eq. 10-15) ---
    
    # Eq (13): Each surgery used at most once 
    for i in valid_surgery_ids:
        if i == 0: continue
        bool_vars = []
        for p in range(num_positions):
            b = model.NewBoolVar(f'b_{i}_{p}')
            model.Add(W[p] == i).OnlyEnforceIf(b)
            model.Add(W[p] != i).OnlyEnforceIf(b.Not())
            bool_vars.append(b)
        model.AddAtMostOne(bool_vars)
        
    # Eq (10): If a position is empty, all later positions are empty 
    for p in range(num_positions - 1):
        is_W_p_zero = model.NewBoolVar(f'is_W_{p}_zero')
        model.Add(W[p] == 0).OnlyEnforceIf(is_W_p_zero)
        model.Add(W[p] != 0).OnlyEnforceIf(is_W_p_zero.Not())
        model.Add(W[p+1] == 0).OnlyEnforceIf(is_W_p_zero)

    # Eq (11): Sequencing constraint (Start time + duration + cleaning) 
    for p in range(num_positions - 1):
        cl_p = model.NewIntVar(0, OCT, f'CL_{p}')
        # Flattened 2D lookup: CL_FLAT_LIST[W[p]][W[p+1]]
        idx_var = model.NewIntVar(0, cl_list_size, f'cl_idx_{p}')
        model.Add(idx_var == W[p] * (max_id + 1) + W[p+1])
        model.AddElement(idx_var, CL_FLAT_LIST, cl_p)
        
        # V_p+1 >= V_p + t_p + CL_p
        model.Add(V[p+1] >= V[p] + t_p_vars[p] + cl_p)

    # Eq (14): Surgeon max daily hours 
    for l_name, l_int in surgeon_map.items():
        t_p_l_vars = [model.NewIntVar(0, day_duration, f't_p_{p}_{l_name}') for p in range(num_positions)]
        for p in range(num_positions):
            b_is_surgeon_l = model.NewBoolVar(f'b_{p}_{l_name}')
            model.Add(s_p_vars[p] == l_int).OnlyEnforceIf(b_is_surgeon_l)
            model.Add(s_p_vars[p] != l_int).OnlyEnforceIf(b_is_surgeon_l.Not())
            
            # Add this position's duration *if* surgeon matches
            model.Add(t_p_l_vars[p] == t_p_vars[p]).OnlyEnforceIf(b_is_surgeon_l)
            model.Add(t_p_l_vars[p] == 0).OnlyEnforceIf(b_is_surgeon_l.Not())
        
        # Sum of durations for this surgeon <= daily limit
        model.Add(sum(t_p_l_vars) <= A_ld[(l_name, day)])
        
    # Eq (15): Start time for empty positions 
    for p in range(1, num_positions):
        is_W_p_zero = model.NewBoolVar(f'is_W_{p}_zero_for_V')
        model.Add(W[p] == 0).OnlyEnforceIf(is_W_p_zero)
        model.Add(W[p] != 0).OnlyEnforceIf(is_W_p_zero.Not())
        
        # V_p = V_{p-1} + t_{p-1}
        model.Add(V[p] == V[p-1] + t_p_vars[p-1]).OnlyEnforceIf(is_W_p_zero)
        
    # Additional Constraint: All surgeries must finish within the day
    for p in range(num_positions):
        model.Add(V[p] + t_p_vars[p] <= day_duration)

    # --- 4. DEFINE OBJECTIVE FUNCTION (Eq. 18) ---
    G_p_vars = [model.NewIntVar(-10000 * SCALING_FACTOR, 10000 * SCALING_FACTOR, f'G_p_{p}') for p in range(num_positions)]
    pi_star_p_vars = [model.NewIntVar(-10000 * SCALING_FACTOR, 10000 * SCALING_FACTOR, f'pi_star_p_{p}') for p in range(num_positions)]
    
    for p in range(num_positions):
        # G_p = G_di_list[W[p]]
        model.AddElement(W[p], G_di_list, G_p_vars[p])
        
        # pi_star_p = PI_STAR_FLAT_LIST[W[p]][V[p]]
        pi_star_idx = model.NewIntVar(0, pi_star_list_size, f'pi_idx_{p}')
        model.Add(pi_star_idx == W[p] * (day_duration + 1) + V[p])
        model.AddElement(pi_star_idx, PI_STAR_FLAT_LIST, pi_star_p_vars[p])

    # Maximize: sum(G_p) - sum(pi_star_p) - pi_d_cost 
    model.Maximize(
        sum(G_p_vars) - 
        sum(pi_star_p_vars) - 
        int(pi_d_cost * SCALING_FACTOR)
    )
    
    # --- 5. SOLVE & PARSE ---
    solver = cp_model.CpSolver()
    # Set a time limit for the subproblem solve
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # We need to un-scale the objective value to get the true reduced cost
        reduced_cost = solver.ObjectiveValue() / SCALING_FACTOR
        
        # Eq (19): Only generate columns with non-negative reduced cost 
        if reduced_cost > 1e-6: # Use a small tolerance
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
                    
                    # Add to surgeon total work
                    new_schedule["surgeon_work"][surg_name] = \
                        new_schedule["surgeon_work"].get(surg_name, 0) + surg_data["duration"]
                    
                    # Add busy times for the master problem's overlap constraint
                    for t_busy in SIMPLIFIED_TIMES:
                         if start_time <= t_busy < start_time + surg_data["duration"]:
                                new_schedule["surgeon_busy_times"][(surg_name, day, t_busy)] = 1
                                
            print(f"  ...FOUND new schedule {new_schedule['id']} with RC = {reduced_cost:.4f}")
            return new_schedule, reduced_cost

    print(f"  ...No profitable new schedule found for {day}.")
    return None, 0.0

if __name__ == "__main__":
    
    known_schedules = get_initial_schedules()
    iteration = 0
    
    while True:
        iteration += 1
        print(f"\n--- COLUMN GENERATION: ITERATION {iteration} ---")
        
        # --- 1. Solve the Master Problem (LP) ---
        print(f"Solving Master Problem (LP) with {len(known_schedules)} schedules...")
        master_lp = build_master_lp(known_schedules, MANDATORY_SURGERIES, OPTIONAL_SURGERIES,
                                    ALL_SURGEONS, ALL_DAYS, ALL_TIMES, K_d, A_ld)
        master_lp.solve(pulp.PULP_CBC_CMD(msg=0)) # msg=0 silences solver output
        
        if master_lp.status != pulp.LpStatusOptimal:
            print("Master LP failed to solve. Stopping.")
            break
            
        # --- 2. Get Dual Prices (pi) ---
        dual_prices = extract_dual_prices(master_lp)
        
        # --- 3. Solve Subproblems (one for each day) ---
        new_schedules_found = False
        for day in ALL_DAYS:
            # --- THIS IS THE REAL SUBPROBLEM CALL ---
            new_schedule, reduced_cost = solve_subproblem(
                day, dual_prices, ALL_SURGERIES_DATA, 
                ALL_SURGEONS, ALL_TIMES, A_ld
            )
            # ----------------------------------------
            
            if reduced_cost > 1e-6:
                new_schedules_found = True
                # Add to list if it's truly new (should be, due to random id)
                if not any(s['id'] == new_schedule['id'] for s in known_schedules):
                    known_schedules.append(new_schedule)
                        
        # --- 4. Check for Convergence ---
        if not new_schedules_found:
            print("\n--- CONVERGENCE REACHED ---")
            print("Subproblems found no new profitable schedules.")
            print("Column Generation loop is finished.")
            break
        
        if iteration > 5:
            print("\nReached max iterations for example. Stopping.")
            break

    # --- 5. FINAL INTEGER SOLVE ---
    print(f"\nBuilding FINAL problem with {len(known_schedules)} total schedules...")
    final_prob = build_master_lp(known_schedules, MANDATORY_SURGERIES, OPTIONAL_SURGERIES,
                                ALL_SURGEONS, ALL_DAYS, ALL_TIMES, K_d, A_ld)
    
    for v in final_prob.variables():
        v.cat = 'Integer'
        v.upBound = 1
        v.lowBound = 0

    print("Solving the FINAL Integer Problem for the real answer...")
    final_prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # --- 6. PRINT THE REAL ANSWER ---
    print(f"\n--- FINAL SOLUTION ---")
    print(f"Status: {pulp.LpStatus[final_prob.status]}")
    
    if final_prob.status == pulp.LpStatusOptimal:
        print("\n--- FINAL OR ROOM ALLOCATION ---")
        total_time = 0
        for v in final_prob.variables():
            if v.value() > 0.9:
                print(f"  USE SCHEDULE: {v.name}")
                for s in known_schedules:
                    if s["id"] == v.name.replace("Schedule_", ""):
                        print(f"    - Day: {s['day']}, Surgeries: {s['surgeries']}, Time: {s['B_j']} min")
                        total_time += s['B_j']
        print(f"\nTotal Scheduled Time: {total_time} minutes")
    else:
        print("\nNo valid integer schedule allocation found.")