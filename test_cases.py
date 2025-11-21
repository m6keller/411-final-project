import random
import os
import csv
from run_optimization_scenario import run_optimization_scenario
from surgery_generation import generate_surgery_data
from visualizer import visualize_schedule, create_summary_visualizations

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
            "infection_type": surg.infection_type,
            "surgery_object": surg
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

def main():
    """
    Defines and runs the test suite for analyzing the impact of different resource levels.
    """
    random.seed(42) # Use a fixed seed for reproducibility
    
    # --- Define Experiment Parameters ---
    N_SURGERIES = 50
    SURGEON_COUNTS = [3, 5, 8]
    DAY_COUNTS = [5, 6, 7, 8]
    OR_COUNTS = [3, 5, 8]
    
    # --- Setup Directories ---
    ANALYSIS_DIR = "analysis"
    OVERALL_DIR = os.path.join(ANALYSIS_DIR, "overall")
    os.makedirs(OVERALL_DIR, exist_ok=True)
    
    all_results_data = []
    
    # --- Run All Combinations ---
    for surgeons in SURGEON_COUNTS:
        for days in DAY_COUNTS:
            for ors in OR_COUNTS:
                scenario_name = f"surg_{surgeons}_days_{days}_ors_{ors}"
                print(f"--- Running Scenario: {scenario_name} ---")

                # 1. Generate the scenario data
                scenario_params = setup_scenario(
                    num_surgeries=N_SURGERIES,
                    num_surgeons=surgeons,
                    num_days=days,
                    num_ors=ors
                )
                
                # 2. Run the optimization
                result = run_optimization_scenario(
                    scenario_name=scenario_name,
                    **scenario_params
                )
                
                # 3. Save the Gantt chart visualization
                gantt_path = os.path.join(ANALYSIS_DIR, f"{scenario_name}.png")
                visualize_schedule(result, scenario_params, gantt_path)
                
                # 4. Collect data for CSV report
                all_surgery_ids = set(scenario_params['all_surgeries_data'].keys())
                scheduled_surgery_ids = set()
                if result.get("selected_schedules"):
                    for schedule in result["selected_schedules"]:
                        for surg_obj in schedule.surgeries_data:
                            scheduled_surgery_ids.add(surg_obj.id)
                
                unscheduled_surgeries = list(all_surgery_ids - scheduled_surgery_ids)
                
                unscheduled_reason = "Not selected by optimizer (due to resource constraints or optionality)"
                
                all_results_data.append({
                    "scenario_name": scenario_name,
                    "total_surgeries": N_SURGERIES,
                    "num_surgeons": surgeons,
                    "num_days": days,
                    "num_ors": ors,
                    "successful_surgeries": len(scheduled_surgery_ids),
                    "unscheduled_surgeries_ids": unscheduled_surgeries,
                    "unscheduled_reason": unscheduled_reason if unscheduled_surgeries else "N/A",
                    "runtime_sec": result['runtime_sec'],
                    "optimization_status": result['status']
                })

    # --- Write Results to CSV in the 'overall' directory ---
    csv_path = os.path.join(OVERALL_DIR, "experiment_summary.csv")
    headers = [
        "scenario_name", "total_surgeries", "num_surgeons", "num_days", "num_ors", 
        "successful_surgeries", "unscheduled_surgeries_ids", "unscheduled_reason", 
        "runtime_sec", "optimization_status"
    ]
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_results_data)
        
    print(f"\nSaved detailed experiment results to {csv_path}")
    
    # --- Generate Summary Visualizations in the 'overall' directory ---
    create_summary_visualizations(csv_path)
    
    print("\n--- Experiment Suite Finished ---")


if __name__ == "__main__":
    main()
