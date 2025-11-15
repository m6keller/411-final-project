import pulp
import random


from ortools.sat.python import cp_model

# --- 1. FAKE DATA (To make the example runnable) ---
# Master list of ALL surgeries the subproblem can choose from
ALL_SURGERIES_DATA = {
    101: {"duration": 120, "surgeon": "Dr_A", "deadline": 3},
    102: {"duration": 120, "surgeon": "Dr_B", "deadline": 3},
    103: {"duration": 300, "surgeon": "Dr_A", "deadline": 1},
    104: {"duration": 60,  "surgeon": "Dr_A", "deadline": 5},
    105: {"duration": 180, "surgeon": "Dr_B", "deadline": 5},
}


# --- FAKE Problem Parameters ---
MANDATORY_SURGERIES = [101, 102, 103]
OPTIONAL_SURGERIES = [104, 105]
ALL_SURGEONS = ["Dr_A", "Dr_B"]
ALL_DAYS = ["Mon", "Tue"]
# Assuming 8-hour day, 9am to 5pm (480 minutes)
# We can use minutes as our "time slots" for this example
ALL_TIMES = list(range(0, 480)) 
K_d = {"Mon": 5, "Tue": 5}
A_ld = {
    ("Dr_A", "Mon"): 480, ("Dr_B", "Mon"): 480,
    ("Dr_A", "Tue"): 480, ("Dr_B", "Tue"): 480,
}

# --- 2. MASTER PROBLEM FUNCTION ---
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
    prob += pulp.lpSum(
        s["B_j"] * sched_vars[s["id"]] for s in known_schedules
    ), "Total_Surgery_Time"
    for i in mandatory_surgeries:
        prob += pulp.lpSum(
            sched_vars[s["id"]] for s in known_schedules if i in s["surgeries"]
        ) == 1, f"Pi_i_Mandatory_{i}"
    for i in optional_surgeries:
        prob += pulp.lpSum(
            sched_vars[s["id"]] for s in known_schedules if i in s["surgeries"]
        ) <= 1, f"Pi_i_Optional_{i}"
    for d in all_days:
        prob += pulp.lpSum(
            sched_vars[s["id"]] for s in known_schedules if s["day"] == d
        ) <= K_d[d], f"Pi_d_OR_Limit_{d}"
    for l in all_surgeons:
        for d in all_days:
            prob += pulp.lpSum(
                s["surgeon_work"].get(l, 0) * sched_vars[s["id"]]
                for s in known_schedules if s["day"] == d
            ) <= A_ld[(l, d)], f"Pi_ld_Surgeon_Hours_{l}_{d}"
            
    # Simplified time: check every 30 minutes instead of every minute
    # A full model would use every time slot (e.g., ALL_TIMES)
    simplified_times = list(range(0, 480, 30))
    for l in all_surgeons:
        for d in all_days:
            for t in simplified_times:
                prob += pulp.lpSum(
                    s["surgeon_busy_times"].get((l, d, t), 0) * sched_vars[s["id"]]
                    for s in known_schedules if s["day"] == d
                ) <= 1, f"Pi_ldt_Surgeon_Overlap_{l}_{d}_{t}"
    return prob

# --- 3. HELPER FUNCTIONS ---

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

# --- 4. REAL SUBPROBLEM IMPLEMENTATION ---

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
    
    # --- 1. PREPROCESSING (Calculate Lookup Tables, Eq. 17) ---
    G_di = {0: 0} # 0 is the "null" surgery
    for i, surg in all_surgeries_data.items():
        pi_i = (
            dual_prices.get(f"Pi_i_Mandatory_{i}", 0) + 
            dual_prices.get(f"Pi_i_Optional_{i}", 0)
        )
        pi_ld = dual_prices.get(f"Pi_ld_Surgeon_Hours_{surg['surgeon']}_{day}", 0)
        t_i = surg['duration']
        G_di[i] = t_i - pi_i - (t_i * pi_ld)

    # Simplified pi_dit_star (start-time cost)
    # A real implementation would be a complex 2D lookup.
    # We will ignore this term for the simplified objective.
    
    pi_d_cost = dual_prices.get(f"Pi_d_OR_Limit_{day}", 0)
    
    # --- 2. MODEL & VARIABLE SETUP ---
    model = cp_model.CpModel()
    num_positions = 8 # Max number of surgeries per OR
    surgery_ids = [0] + list(all_surgeries_data.keys())
    day_duration = all_times[-1] + 1
    
    W = [model.NewIntVarFromDomain(cp_model.Domain.FromValues(surgery_ids), f'W_{p}') 
         for p in range(num_positions)]
    V = [model.NewIntVar(0, day_duration, f'V_{p}') 
         for p in range(num_positions)]

    # --- 3. ADD CONSTRAINTS (Simplified Eq. 10-15) ---
    
    # Eq (13): Each surgery used at most once
    for i in all_surgeries_data.keys():

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

    # --- 4. DEFINE OBJECTIVE FUNCTION (Simplified Eq. 18) ---
    # We use the pre-calculated G_di table.
    # G_di_list is a list where index 'i' holds the profit for surgery 'i'.
    # CP-SAT requires integer coefficients. We scale by 1000 and convert to int.
    SCALING_FACTOR = 1000
    max_id = max(surgery_ids)
    G_di_list = [int(G_di.get(i, 0) * SCALING_FACTOR) for i in range(max_id + 1)]
    
    G_p_vars = [model.NewIntVar(-10000 * SCALING_FACTOR, 10000 * SCALING_FACTOR, f'G_p_{p}') for p in range(num_positions)]
    for p in range(num_positions):
        # This looks up the profit G_di[W[p]] and stores it in G_p_vars[p]
        model.AddElement(W[p], G_di_list, G_p_vars[p])

    # Simplified Objective: Maximize sum(G_p) - pi_d_cost
    # A full implementation would also subtract the complex pi_dit_star costs.
    model.Maximize(sum(G_p_vars) - int(pi_d_cost * SCALING_FACTOR))
    
    # --- 5. SOLVE & PARSE ---
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        # We need to un-scale the objective value to get the true reduced cost
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
                    surg = surg_data["surgeon"]
                    new_schedule["surgeon_work"][surg] = \
                        new_schedule["surgeon_work"].get(surg, 0) + surg_data["duration"]
                    
                    # Add busy times (simplified)
                    for t_busy in range(start_time, start_time + surg_data["duration"], 30):
                         if t_busy in all_times:
                            new_schedule["surgeon_busy_times"][(surg, day, t_busy)] = 1
            
            print(f"  ...FOUND new schedule {new_schedule['id']} with RC > 0.")
            return new_schedule, reduced_cost

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
            # --- THIS IS THE CHANGE ---
            # Call the REAL subproblem, not the stub
            new_schedule, reduced_cost = solve_subproblem(
                day, dual_prices, ALL_SURGERIES_DATA, 
                ALL_SURGEONS, ALL_TIMES, A_ld
            )
            # --------------------------
            
            if reduced_cost > 1e-6:
                new_schedules_found = True
                if new_schedule not in known_schedules:
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