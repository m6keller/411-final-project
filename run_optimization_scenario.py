import pulp
import random
import time
import concurrent.futures

from schedule import Schedule
from subproblem_optimizer import generate_daily_schedule
from surgery_generation import generate_surgery_data

PRINT_MULTIPLIER = 90
BIG_M = 100000  # Penalty for failing to schedule a mandatory surgery

def build_master_lp(known_schedules, mandatory_surgeries, optional_surgeries,
                    all_surgeons, all_days, K_d, A_ld, SIMPLIFIED_TIMES):
    """
    Master Problem with Artificial Variables to ensure feasibility.
    """
    prob = pulp.LpProblem("Master_Problem_LP", pulp.LpMaximize)
    
    # Decision Variables for Schedules
    sched_vars = pulp.LpVariable.dicts(
        "Schedule",
        [s.id for s in known_schedules],
        lowBound=0,
        cat='Continuous'
    )
    
    # Artificial Variables for Mandatory Surgeries (Slack Variables)
    # If art_var_i = 1, it means surgery i was NOT scheduled.
    art_vars = pulp.LpVariable.dicts(
        "Art_Mandatory", 
        mandatory_surgeries, 
        lowBound=0, 
        upBound=1, 
        cat='Continuous'
    )

    # (1) Objective: Maximize Schedule Duration - Big M Penalties
    prob += (
        pulp.lpSum(s.B_j * sched_vars[s.id] for s in known_schedules) - 
        pulp.lpSum(BIG_M * art_vars[i] for i in mandatory_surgeries)
    ), "Total_Objective"

    # (2) Mandatory Surgeries: Sum(Schedule) + Artificial = 1
    for i in mandatory_surgeries:
        prob += (
            pulp.lpSum(sched_vars[s.id] for s in known_schedules if i in s.surgeries) 
            + art_vars[i] 
            == 1
        ), f"Pi_i_Mandatory_{i}"

    # (3) Optional Surgeries: Sum(Schedule) <= 1
    for i in optional_surgeries:
        prob += (
            pulp.lpSum(sched_vars[s.id] for s in known_schedules if i in s.surgeries) 
            <= 1
        ), f"Pi_i_Optional_{i}"

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
            
    # (6) Surgeon Overlap 
    for l in all_surgeons:
        for d in all_days:
            for t in SIMPLIFIED_TIMES:
                prob += pulp.lpSum(
                    s.surgeon_busy_times.get((l, d, t), 0) * sched_vars[s.id]
                    for s in known_schedules if s.day == d
                ) <= 1, f"Pi_ldt_Surgeon_Overlap_{l}_{d}_{t}"
                
    return prob

    
def get_initial_schedules(all_surgeries_data, mandatory_surgeries, all_days, 
                            DAY_MAP, K_d, A_ld, all_surgeons, OBLIGATORY_CLEANING_TIME):
    """
    Constructive Heuristic (Parallel/Bin Packing).
    Now strictly enforces DEADLINES.
    """
    print("  Generating Initial Heuristic Schedules...")
    
    sorted_surgeries = []
    for s_id in mandatory_surgeries:
        surg_obj = all_surgeries_data[s_id]["surgery_object"]
        sorted_surgeries.append(surg_obj)
    
    # [cite_start]Sort: Earliest Deadline First (EDF), then Longest Duration [cite: 593-595]
    # This helps ensure hard-to-fit deadlines get the first slots.
    sorted_surgeries.sort(key=lambda s: (s.deadline, -s.duration))

    # Trackers
    rooms_registry = {day: [] for day in all_days}
    surgeon_availability = {(s, d): [] for s in all_surgeons for d in all_days}
    surgeon_daily_work = {(s, d): 0 for s in all_surgeons for d in all_days}

    initial_schedules = []

    for surg in sorted_surgeries:
        is_scheduled = False
        
        for day in all_days:
            if is_scheduled: break
            
            current_day_num = DAY_MAP[day]
            if current_day_num > surg.deadline:
                continue
            
            existing_rooms = rooms_registry[day]
            current_surgeon_work = surgeon_daily_work[(surg.surgeon, day)]

            # STRATEGY A: Open NEW Room
            if len(existing_rooms) < K_d[day]:
                start_time = 0
                end_time = surg.duration
                
                overlap = False
                for (busy_start, busy_end) in surgeon_availability[(surg.surgeon, day)]:
                     if max(start_time, busy_start) < min(end_time, busy_end):
                        overlap = True
                        break
                
                valid_work = (current_surgeon_work + surg.duration <= A_ld[(surg.surgeon, day)])

                if not overlap and valid_work:
                    new_room = {
                        'end_time': end_time,
                        'surgeries': [surg.id],
                        'surgeries_data': [surg],
                        'work_log': {surg.surgeon: surg.duration},
                        'busy_log': {(surg.surgeon, start_time): surg.duration},
                        'start_times': {surg.id: start_time}
                    }
                    rooms_registry[day].append(new_room)
                    surgeon_availability[(surg.surgeon, day)].append((start_time, end_time))
                    surgeon_daily_work[(surg.surgeon, day)] += surg.duration
                    is_scheduled = True
                    break

            # STRATEGY B: Pack EXISTING Room
            if not is_scheduled:
                for room in existing_rooms:
                    # Cleaning Logic
                    last_surg = room['surgeries_data'][-1]
                    cleaning_time = 0
                    if last_surg.infection_type > 0 and surg.infection_type == 0:
                        cleaning_time = OBLIGATORY_CLEANING_TIME
                    elif (last_surg.infection_type > 0 and surg.infection_type > 0 
                          and last_surg.infection_type != surg.infection_type):
                        cleaning_time = OBLIGATORY_CLEANING_TIME
                    
                    start_time = room['end_time'] + cleaning_time
                    end_time = start_time + surg.duration
                    
                    if end_time > 480: continue
                    if current_surgeon_work + surg.duration > A_ld[(surg.surgeon, day)]: continue

                    overlap = False
                    for (busy_start, busy_end) in surgeon_availability[(surg.surgeon, day)]:
                        if max(start_time, busy_start) < min(end_time, busy_end):
                            overlap = True
                            break
                    if overlap: continue
                    
                    room['surgeries'].append(surg.id)
                    room['surgeries_data'].append(surg)
                    room['end_time'] = end_time
                    room['work_log'][surg.surgeon] = room['work_log'].get(surg.surgeon, 0) + surg.duration
                    room['busy_log'][(surg.surgeon, start_time)] = surg.duration
                    room['start_times'][surg.id] = start_time
                    
                    surgeon_availability[(surg.surgeon, day)].append((start_time, end_time))
                    surgeon_daily_work[(surg.surgeon, day)] += surg.duration
                    is_scheduled = True
                    break

        if not is_scheduled:
            print(f"WARNING: Heuristic failed to schedule surgery {surg.id} (DL: Day {surg.deadline})")

    for day, rooms in rooms_registry.items():
        for i, room in enumerate(rooms):
            total_duration = sum(room['work_log'].values())
            sched_obj = Schedule(
                id=f"Init_{day}_Room{i+1}",
                B_j=total_duration,
                day=day,
                surgeries=room['surgeries'],
                surgeries_data=room['surgeries_data'],
                surgeon_work=room['work_log'],
                surgeon_busy_times=room['busy_log'],
                start_times=room['start_times']
            )
            initial_schedules.append(sched_obj)

    return initial_schedules


def extract_dual_prices(prob):
    """
    Extracts all dual prices (pi) from the solved master LP.
    """
    dual_prices = {}
    for name, c in prob.constraints.items():
        dual_prices[name] = c.pi
    return dual_prices


def run_optimization_scenario(
    scenario_name, all_surgeries_data, mandatory_surgeries, optional_surgeries,
    all_surgeons, all_days, DAY_MAP, K_d, A_ld, ALL_TIMES, OBLIGATORY_CLEANING_TIME, SIMPLIFIED_TIMES
):
    print(f"\n--- RUNNING SCENARIO: {scenario_name} ---")
    start_time = time.time()
    
    # 1. Initialize with Heuristic (Likely Infeasible, that's okay now)
    from run_optimization_scenario import get_initial_schedules # Import heuristic from existing file
    known_schedules = get_initial_schedules(
        all_surgeries_data, mandatory_surgeries, all_days, 
        DAY_MAP, K_d, A_ld, all_surgeons, OBLIGATORY_CLEANING_TIME
    )
    
    iteration = 0
    MAX_ITER = 15
    
    while iteration < MAX_ITER:
        iteration += 1
        print(f"  Iteration {iteration}...", end=" ")
        
        # 2. Solve RMP
        master_lp = build_master_lp(known_schedules, mandatory_surgeries, optional_surgeries,
                                    all_surgeons, all_days, K_d, A_ld, SIMPLIFIED_TIMES)
        # Use warmStart=False to force re-calc, avoiding stuck solutions
        master_lp.solve(pulp.PULP_CBC_CMD(msg=0, warmStart=False))
        
        if master_lp.status != pulp.LpStatusOptimal:
            print(f"Master LP Failed: {pulp.LpStatus[master_lp.status]}")
            break

        # 3. Extract Duals
        duals = extract_dual_prices(master_lp)
        
        # DEBUG: Print top 3 duals to verify Big M is active
        # Huge negative duals = The constraint is "begging" to be satisfied
        sorted_duals = sorted(duals.items(), key=lambda x: x[1])
        print(f"Top Penalty Duals: {[f'{k}:{v:.1f}' for k,v in sorted_duals[:2]]}")

        # 4. Solve Subproblems (Parallel)
        new_cols_found = 0
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = {
                executor.submit(generate_daily_schedule, day, duals, all_surgeries_data, 
                                all_surgeons, ALL_TIMES, A_ld, DAY_MAP, 
                                OBLIGATORY_CLEANING_TIME, SIMPLIFIED_TIMES): day 
                for day in all_days
            }
            
            for future in concurrent.futures.as_completed(futures):
                new_sched, red_cost = future.result()
                # Lower threshold slightly to catch float errors
                if new_sched and red_cost > 1e-4:
                    # Check for duplicates based on surgery ID content, not just schedule ID
                    existing_signatures = [set(s.surgeries) for s in known_schedules if s.day == new_sched.day]
                    if set(new_sched.surgeries) not in existing_signatures:
                        known_schedules.append(new_sched)
                        new_cols_found += 1

        print(f"Found {new_cols_found} new schedules.")
        if new_cols_found == 0:
            print("  CONVERGENCE REACHED.")
            break

    # --- FINAL INTEGER SOLVE ---
    print("  Solving Final Integer Problem...")
    final_prob = build_master_lp(known_schedules, mandatory_surgeries, optional_surgeries,
                                all_surgeons, all_days, K_d, A_ld, SIMPLIFIED_TIMES)
    
    for v in final_prob.variables():
        v.cat = 'Integer'
    
    final_prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # REPORTING
    print(f"Final Status: {pulp.LpStatus[final_prob.status]}")
    print(f"Final Objective: {pulp.value(final_prob.objective)}")
    
    selected = []
    if final_prob.status == pulp.LpStatusOptimal:
        for v in final_prob.variables():
            if v.name.startswith("Schedule_") and v.value() > 0.9:
                s_id = v.name.replace("Schedule_", "")
                for s in known_schedules:
                    if s.id == s_id: selected.append(s)

    return {
        "scenario_name": scenario_name,
        "status": pulp.LpStatus[final_prob.status],
        "runtime_sec": time.time() - start_time,
        "total_iterations": iteration,
        "total_columns_generated": len(known_schedules),
        "total_scheduled_time": pulp.value(final_prob.objective),
        "selected_schedules": selected,
        "all_days": all_days,
        "all_surgeries_data": all_surgeries_data  
    }
    
def print_results_report(results):
    """
    Prints a detailed, debug-friendly report of the optimization results.
    Now includes a SPECIFIC section for Unscheduled Surgeries to debug failures.
    """
    def fmt_time(minutes_from_start):
        start_hour = 9
        h = start_hour + int(minutes_from_start // 60)
        m = int(minutes_from_start % 60)
        return f"{h:02d}:{m:02d}"

    # 1. Collect all scheduled IDs
    scheduled_ids = set()
    if results['selected_schedules']:
        for sched in results['selected_schedules']:
            for s_id in sched.surgeries:
                scheduled_ids.add(s_id)

    # 2. Identify Missing Surgeries
    # We need access to the original surgery data. 
    # Note: In your main script, ensure 'all_surgeries_data' is passed to this report 
    # or available globally. For this snippet, I assume you pass it or we find it via logic.
    # (In the context of your script, you might need to pass 'all_surgeries_data' into this function)
    
    # HACK: If you didn't pass all_surgeries_data to results, we can't list details. 
    # Assuming you can modify the call to print_results_report(results, all_surgeries_data)
    # For now, I will simulate the report logic assuming we have the data.
    
    print("\n\n" + "="*PRINT_MULTIPLIER)
    print(f"{'OPTIMIZATION DEBUG REPORT':^{PRINT_MULTIPLIER}}")
    print("="*PRINT_MULTIPLIER + "\n")
    
    res = results
    print(f"Scenario:    {res['scenario_name']}")
    print(f"Status:      {res['status']}")
    print(f"Objective:   {res['total_scheduled_time']:.0f} (Negative = Penalties Applied)")
    print("-" * PRINT_MULTIPLIER)

    # --- NEW SECTION: UNSCHEDULED SURGERIES ---
    # You will need to pass 'all_surgeries_data' to this function call in the main block
    if 'all_surgeries_data' in res:
        all_data = res['all_surgeries_data']
        missing_ids = [i for i in all_data.keys() if i not in scheduled_ids and i != 0]
        
        if missing_ids:
            print(f"\n{'!!! UNSCHEDULED SURGERIES (The Reason for Negative Score) !!!':^{PRINT_MULTIPLIER}}")
            print("-" * PRINT_MULTIPLIER)
            print(f"   {'ID':<5} | {'SURGEON':<10} | {'DUR':<5} | {'DEADLINE':<10} | {'REASON HYPOTHESIS'}")
            print("   " + "."*85)
            
            for m_id in missing_ids:
                s_obj = all_data[m_id]['surgery_object']
                
                # Logic to guess why it failed
                reason = "Unknown"
                if s_obj.deadline < len(res['all_days']):
                    reason = f"Deadline Day {s_obj.deadline} (Expired before Day {len(res['all_days'])})"
                else:
                    reason = "Capacity/Surgeon Constraints"
                
                print(f"   {m_id:<5} | {s_obj.surgeon:<10} | {s_obj.duration:<5} | Day {s_obj.deadline:<5} | {reason}")
            print("-" * PRINT_MULTIPLIER)
            print(f"   * These surgeries could not fit before their deadline.")
    # -------------------------------------------

    schedules_by_day = {day: [] for day in res['all_days']}
    if res['selected_schedules']:
        for sched in res['selected_schedules']:
            schedules_by_day[sched.day].append(sched)

    print(f"\n{'DAILY SCHEDULE BREAKDOWN':^{PRINT_MULTIPLIER}}")
    print("="*PRINT_MULTIPLIER)

    total_surgeries_count = 0

    for day in res['all_days']:
        day_schedules = schedules_by_day[day]
        print(f"\nDay: {day}")
        print("-" * PRINT_MULTIPLIER)

        if not day_schedules:
            print("   [No Operational Rooms Scheduled]")
            continue

        day_schedules.sort(key=lambda x: x.id)

        for i, sched in enumerate(day_schedules):
            util_pct = (sched.B_j / 480) * 100
            print(f"   OR #{i+1} | Load: {sched.B_j}m ({util_pct:.1f}%)")
            
            sorted_surgeries = sorted(
                sched.surgeries_data, 
                key=lambda s: sched.start_times.get(s.id, 0)
            )

            print(f"   {'TIME':<13} | {'ID':<3} | {'SURGEON':<10} | {'DL':<3} | {'DUR':<5}")
            print("   " + "."*50)

            for s in sorted_surgeries:
                start_min = sched.start_times.get(s.id, 0)
                end_min = start_min + s.duration
                total_surgeries_count += 1
                print(f"   {fmt_time(start_min)} - {fmt_time(end_min)} | {s.id:<3} | {s.surgeon:<10} | {s.deadline:<3} | {s.duration}m")
            print("")
            
    print("="*PRINT_MULTIPLIER)
    print(f"Total Surgeries Scheduled: {total_surgeries_count}")
    print("="*PRINT_MULTIPLIER + "\n")
         

if __name__ == "__main__":
    
    random.seed(42) # Set seed for reproducible results

    # --- Configuration ---
    NUM_SURGERIES = 50
    NUM_SURGEONS = 5
    NUM_DAYS = 6
    NUM_ORS = 3
    
    # --- Constants ---
    OBLIGATORY_CLEANING_TIME = 30
    ALL_TIMES = list(range(0, 480)) # 8-hour day
    SIMPLIFIED_TIMES = list(range(0, 480, OBLIGATORY_CLEANING_TIME))
    
    # --- Generate Resources ---
    all_surgeons = [f"Surgeon_{chr(65+i)}" for i in range(NUM_SURGEONS)]
    all_days = [f"Day_{i+1}" for i in range(NUM_DAYS)]
    DAY_MAP = {day: i+1 for i, day in enumerate(all_days)}
    
    # --- OR capacities ---
    K_d = {day: NUM_ORS for day in all_days}

    # How many minutes each surgeon can work each day
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
            "infection_type": surg.infection_type,
            "surgery_object": surg
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