import random
import time
from run_optimization_scenario import run_optimization_scenario, print_results_report
from surgery_generation import generate_surgery_data

def setup_scenario(num_surgeries, num_surgeons, num_days, num_ors):
    """
    Generates a complete set of scenario parameters based on the inputs.
    """
    # --- Constants ---
    OBLIGATORY_CLEANING_TIME = 30
    ALL_TIMES = list(range(0, 480)) # 8-hour day
    SIMPLIFIED_TIMES = list(range(0, 480, OBLIGATORY_CLEANING_TIME))
    
    # --- Generate Resources ---
    all_surgeons = [f"Surgeon_{chr(65+i)}" for i in range(num_surgeons)]
    all_days = [f"Day_{i+1}" for i in range(num_days)]
    DAY_MAP = {day: i+1 for i, day in enumerate(all_days)}
    
    K_d = {day: num_ors for day in all_days}
    A_ld = {(surg, day): 480 for surg in all_surgeons for day in all_days}

    # --- Generate Surgeries ---
    all_surgeries_data = {}
    mandatory_surgeries = []
    optional_surgeries = []
    
    generated_surgeries = generate_surgery_data(num_surgeries, num_days, num_surgeons)
    for surg in generated_surgeries:
        surg_id = surg.id
        all_surgeries_data[surg_id] = {
            "duration": surg.duration,
            "surgeon": surg.surgeon,
            "deadline": surg.deadline,
            "infection_type": surg.infection_type
        }
        
        if surg.deadline <= num_days:
            mandatory_surgeries.append(surg_id)
        else:
            optional_surgeries.append(surg_id)
            
    return {
        "all_surgeries_data": all_surgeries_data,
        "mandatory_surgeries": mandatory_surgeries,
        "optional_surgeries": optional_surgeries,
        "all_surgeons": all_surgeons,
        "all_days": all_days,
        "DAY_MAP": DAY_MAP,
        "K_d": K_d,
        "A_ld": A_ld,
        "ALL_TIMES": ALL_TIMES,
        "OBLIGATORY_CLEANING_TIME": OBLIGATORY_CLEANING_TIME,
        "SIMPLIFIED_TIMES": SIMPLIFIED_TIMES
    }

def print_summary_report(all_results):
    """
    Prints a clear comparison report from all scenario results.
    """
    print("\n\n" + "="*80)
    print(" " * 28 + "OPTIMIZATION TEST SUITE REPORT")
    print("="*80 + "\n")
    
    # --- Print Summary Table ---
    print(f"{ 'SCENARIO':<15} | { 'STATUS':<26} | { 'TIME (s)':<8} | { 'CG ITERS':<8} | { 'COLUMNS':<7} | { 'TOTAL MINS':<10}")
    print("-"*80)
    for res in all_results:
        print(f"{res['scenario_name']:<15} | {res['status']:<26} | {res['runtime_sec']:<8.2f} | {res['total_iterations']:<8} | {res['total_columns_generated']:<7} | {res['total_scheduled_time']:<10.0f}")
        
    print("\n" + "="*80)
    print(" " * 28 + "DETAILED OPTIMAL OUTPUTS")
    print("="*80 + "\n")

    # --- Print Detailed Schedules ---
    for res in all_results:
        print_results_report(res)


def main():
    """
    Defines and runs the test suite.
    """
    random.seed(1) # Set seed for reproducible results

    scenarios_to_run = [
        {"name": "Relaxed", "surgeries": 5, "surgeons": 5, "days": 5, "ors": 5},
        {"name": "Moderate", "surgeries": 8, "surgeons": 5, "days": 5, "ors": 5},
        {"name": "Busy", "surgeries": 10, "surgeons": 3, "days": 5, "ors": 2},
        {"name": "Super Busy", "surgeries": 8, "surgeons": 2, "days": 4, "ors": 2},
    ]
    
    all_results = []
    
    for scenario in scenarios_to_run:
        # 1. Generate the scenario data
        scenario_params = setup_scenario(
            num_surgeries=scenario["surgeries"],
            num_surgeons=scenario["surgeons"],
            num_days=scenario["days"],
            num_ors=scenario["ors"]
        )
        
        # 2. Run the optimization
        result = run_optimization_scenario(
            scenario_name=scenario["name"],
            **scenario_params
        )
        
        all_results.append(result)
        
    # 3. Print the final summary report
    print_summary_report(all_results)

if __name__ == "__main__":
    main()
