import pulp
import random
import time

from schedule import Schedule
from subproblem_optimizer import generate_daily_schedule
from surgery_generation import generate_surgery_data


# --- Scheduler Core Logic ---

def build_master_lp(known_schedules, mandatory_surgeries, optional_surgeries,
                    all_surgeons, all_days, K_d, A_ld, SIMPLIFIED_TIMES):
    """
    Builds the Master Problem LP model from the paper.
    """
    prob = pulp.LpProblem("Master_Problem_LP", pulp.LpMaximize)
    
    if not known_schedules:
        prob += 0, "Empty_Problem"
        prob.status = pulp.LpStatusOptimal
        return prob

    sched_vars = pulp.LpVariable.dicts(
        "Schedule",
        [s.id for s in known_schedules],
        lowBound=0,
        cat='Continuous'
    )
    
    # (1) Objective function
    prob += pulp.lpSum(
        s.B_j * sched_vars[s.id] for s in known_schedules
    ), "Total_Surgery_Time"

    # (2) Mandatory Surgeries 
    for i in mandatory_surgeries:
        prob += pulp.lpSum(
            sched_vars[s.id] for s in known_schedules if i in s.surgeries
        ) == 1, f"Pi_i_Mandatory_{i}"

    # (3) Optional Surgeries 
    for i in optional_surgeries:
        prob += pulp.lpSum(
            sched_vars[s.id] for s in known_schedules if i in s.surgeries
        ) <= 1, f"Pi_i_Optional_{i}"

    # (4) Daily OR Limit 
    for d in all_days:
        prob += pulp.lpSum(
            sched_vars[s.id] for s in known_schedules if s.day == d
        ) <= K_d[d], f"Pi_d_OR_Limit_{d}"

    # (5) Surgeon Max Hours 
    for l in all_surgeons:
        for d in all_days:
            prob += pulp.lpSum(
                s.surgeon_work.get(l, 0) * sched_vars[s.id]
                for s in known_schedules if s.day == d
            ) <= A_ld[(l, d)], f"Pi_ld_Surgeon_Hours_{l}_{d}"
            
    # (6) Surgeon Overlap (Coloring Constraint) 
    for l in all_surgeons:
        for d in all_days:
            for t in SIMPLIFIED_TIMES:
                prob += pulp.lpSum(
                    s.surgeon_busy_times.get((l, d, t), 0) * sched_vars[s.id]
                    for s in known_schedules if s.day == d
                ) <= 1, f"Pi_ldt_Surgeon_Overlap_{l}_{d}_{t}"
                
    return prob

def get_initial_schedules(all_surgeries_data, mandatory_surgeries, all_days, DAY_MAP, 
                          SIMPLIFIED_TIMES, K_d, A_ld, OBLIGATORY_CLEANING_TIME, ALL_TIMES, all_surgeons):
    """
    Creates a set of mutually feasible initial schedules for mandatory surgeries
    using a greedy heuristic to avoid surgeon and OR conflicts.
    """
    initial_schedules = []
    if not mandatory_surgeries:
        return []

    day_duration = ALL_TIMES[-1] + 1
    
    # Track resource usage for the initial heuristic
    # surgeon_availability: (surgeon, day) -> list of (start, end) busy intervals
    surgeon_availability = {(s, d): [] for s in all_surgeons for d in all_days}
    # or_usage: day -> count of ORs used
    or_usage = {d: 0 for d in all_days}

    # Create a schedule for each mandatory surgery
    for surg_id in mandatory_surgeries:
        surg_data = all_surgeries_data[surg_id]
        surg_name = surg_data["surgeon"]
        duration = surg_data["duration"]
        surg_deadline_day = surg_data["deadline"]
        
        found_slot = False
        
        # Try to find a slot for this surgery
        for day_name in all_days:
            day_num = DAY_MAP[day_name]
            if day_num > surg_deadline_day:
                continue # Past deadline

            # Check for a free OR on this day
            if or_usage[day_name] >= K_d[day_name]:
                continue # No ORs left on this day

            # Try to find a time slot for the surgeon
            # Iterate through possible start times (e.g., every 30 mins)
            for start_time in range(0, day_duration - duration + 1, 30):
                finish_time = start_time + duration
                
                # Check if surgeon's daily max time is exceeded
                current_work = sum(end - start for start, end in surgeon_availability[(surg_name, day_name)])
                if current_work + duration > A_ld[(surg_name, day_name)]:
                    break # Surgeon has no more time on this day, try next day

                # Check for overlap with surgeon's existing schedule on this day
                is_overlap = False
                for busy_start, busy_end in surgeon_availability[(surg_name, day_name)]:
                    if max(start_time, busy_start) < min(finish_time, busy_end):
                        is_overlap = True
                        break
                
                if not is_overlap:
                    # Found a valid slot
                    found_slot = True
                    
                    # --- Create and add the new schedule ---
                    busy_times = {}
                    for t_busy in SIMPLIFIED_TIMES:
                        if start_time <= t_busy < finish_time:
                            busy_times[(surg_name, day_name, t_busy)] = 1
                    
                    schedule = Schedule(
                        schedule_id=f"Initial_Sched_{surg_id}",
                        B_j=duration,
                        day=day_name,
                        surgeries=[surg_id],
                        surgeon_work={surg_name: duration},
                        surgeon_busy_times=busy_times
                    )
                    initial_schedules.append(schedule)
                    
                    # --- Update resource usage ---
                    surgeon_availability[(surg_name, day_name)].append((start_time, finish_time))
                    or_usage[day_name] += 1
                    
                    break # Move to the next surgery
            
            if found_slot:
                break # Move to the next surgery

        if not found_slot:
            print(f"WARNING: Heuristic could not find a feasible slot for mandatory surgery {surg_id}. Creating a conflicting dummy schedule on Day 1.")
            # Create a dummy schedule on Day 1 at time 0. This is likely to cause
            # infeasibility in the Master LP, but ensures the column exists.
            day_name = all_days[0]
            start_time = 0
            finish_time = start_time + duration
            
            busy_times = {}
            for t_busy in SIMPLIFIED_TIMES:
                if start_time <= t_busy < finish_time:
                    busy_times[(surg_name, day_name, t_busy)] = 1

            schedule = Schedule(
                schedule_id=f"Initial_Sched_{surg_id}_DUMMY",
                B_j=duration, day=day_name, surgeries=[surg_id],
                surgeon_work={surg_name: duration}, surgeon_busy_times=busy_times
            )
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
    all_times,
    A_ld,
    DAY_MAP,
    OBLIGATORY_CLEANING_TIME,
    SIMPLIFIED_TIMES
):
    """
    Wrapper for the subproblem solver.
    """
    return generate_daily_schedule(
        day, dual_prices, all_surgeries_data, 
        all_surgeons, all_times, A_ld, DAY_MAP,
        OBLIGATORY_CLEANING_TIME, SIMPLIFIED_TIMES
    )

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
    Runs the full Column Generation and final Integer Solve for a given scenario.
    """
    print(f"\n--- RUNNING SCENARIO: {scenario_name} ---")
    start_time = time.time()
    
    results = {
        "scenario_name": scenario_name,
        "status": "Failed",
        "total_scheduled_time": 0,
        "selected_schedules": [],
        "total_iterations": 0,
        "total_columns_generated": 0,
        "runtime_sec": 0,
        "all_days": all_days
    }

    known_schedules = get_initial_schedules(
        all_surgeries_data, mandatory_surgeries, all_days, DAY_MAP, 
        SIMPLIFIED_TIMES, K_d, A_ld, OBLIGATORY_CLEANING_TIME, ALL_TIMES, all_surgeons
    )
    
    results["total_columns_generated"] = len(known_schedules)
    
    iteration = 0
    max_iterations = 15 
    
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
            return results 
            
        dual_prices = extract_dual_prices(master_lp)
        
        new_schedules_found = False
        for day in all_days:
            new_schedule, reduced_cost = solve_subproblem(
                day, dual_prices, all_surgeries_data, 
                all_surgeons, ALL_TIMES, A_ld, DAY_MAP,
                OBLIGATORY_CLEANING_TIME, SIMPLIFIED_TIMES
            )
            
            if new_schedule and reduced_cost > 1e-6:
                new_schedules_found = True
                if not any(s.id == new_schedule.id for s in known_schedules):
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
    
    results["status"] = pulp.LpStatus[final_prob.status]
    results["runtime_sec"] = end_time - start_time

    if final_prob.status == pulp.LpStatusOptimal:
        results["total_scheduled_time"] = pulp.value(final_prob.objective)
        for v in final_prob.variables():
            if v.value() > 0.9:
                for s in known_schedules:
                    if s.id == v.name.replace("Schedule_", ""):
                        results["selected_schedules"].append(s)
    
    print(f"--- {scenario_name} FINISHED ({(end_time - start_time):.2f}s) ---")
    return results

def print_results_report(results):
    """
    Prints a clear report from a single scenario result.
    """
    print("\n\n" + "="*80)
    print(" " * 28 + "OPTIMIZATION RUN REPORT")
    print("="*80 + "\n")
    
    print(f"{'SCENARIO':<15} | {'STATUS':<26} | {'TIME (s)':<8} | {'CG ITERS':<8} | {'COLUMNS':<7} | {'TOTAL MINS':<10}")
    print("-"*80)
    res = results
    print(f"{res['scenario_name']:<15} | {res['status']:<26} | {res['runtime_sec']:<8.2f} | {res['total_iterations']:<8} | {res['total_columns_generated']:<7} | {res['total_scheduled_time']:<10.0f}")
        
    print("\n" + "="*80)
    print(" " * 30 + "DETAILED OPTIMAL OUTPUT")
    print("="*80 + "\n")

    print(f"\n--- {res['scenario_name']} (Scheduled: {res['total_scheduled_time']:.0f} mins) ---")
    if res['status'] != "Optimal":
        print(f"  No optimal solution found (Status: {res['status']}).")
        return
        
    if not res['selected_schedules']:
        print("  No schedules were selected in the final plan.")
        return

    schedules_by_day = {day: [] for day in res['all_days']}
    for sched in res['selected_schedules']:
        day = sched.day
        if day not in schedules_by_day:
             schedules_by_day[day] = []
        schedules_by_day[day].append(sched)
        
    for day, schedules in schedules_by_day.items():
        if not schedules:
            continue
        print(f"  ORs for {day}:")
        for i, sched in enumerate(schedules):
            print(f"    - OR {i+1}: Surgeries {sched.surgeries} ({sched.B_j} min) [ID: {sched.id}]")
                

if __name__ == "__main__":
    
    random.seed(42) # Set seed for reproducible results

    # --- Configuration ---
    NUM_SURGERIES = 10
    NUM_SURGEONS = 5
    NUM_DAYS = 10
    
    # --- Constants ---
    OBLIGATORY_CLEANING_TIME = 30
    ALL_TIMES = list(range(0, 480)) # 8-hour day
    SIMPLIFIED_TIMES = list(range(0, 480, OBLIGATORY_CLEANING_TIME))
    
    # --- Generate Resources ---
    all_surgeons = [f"Surgeon_{chr(65+i)}" for i in range(NUM_SURGEONS)]
    all_days = [f"Day_{i+1}" for i in range(NUM_DAYS)]
    DAY_MAP = {day: i+1 for i, day in enumerate(all_days)}
    
    K_d = {day: NUM_SURGEONS for day in all_days}
    A_ld = {(surg, day): 480 for surg in all_surgeons for day in all_days}

    # --- Generate Surgeries ---
    all_surgeries_data = {}
    mandatory_surgeries = []
    optional_surgeries = []
    
    # Use surgery_generation script and convert to dict
    generated_surgeries = generate_surgery_data(NUM_SURGERIES, NUM_DAYS, NUM_SURGEONS)
    for surg in generated_surgeries:
        surg_id = surg.id

        all_surgeries_data[surg_id] = {
            "duration": surg.duration,
            "surgeon": surg.surgeon,
            "deadline": surg.deadline,
            "infection_type": surg.infection_type
        }
        
        if surg.deadline <= NUM_DAYS:
            mandatory_surgeries.append(surg_id)
        else:
            optional_surgeries.append(surg_id)

    # --- Run Optimization ---
    result = run_optimization_scenario(
        scenario_name="Dummy_Run",
        all_surgeries_data=all_surgeries_data,
        mandatory_surgeries=mandatory_surgeries,
        optional_surgeries=optional_surgeries,
        all_surgeons=all_surgeons,
        all_days=all_days,
        DAY_MAP=DAY_MAP,
        K_d=K_d,
        A_ld=A_ld,
        ALL_TIMES=ALL_TIMES,
        OBLIGATORY_CLEANING_TIME=OBLIGATORY_CLEANING_TIME,
        SIMPLIFIED_TIMES=SIMPLIFIED_TIMES
    )
    
    # --- Print Report ---
    print_results_report(result)
    
    
