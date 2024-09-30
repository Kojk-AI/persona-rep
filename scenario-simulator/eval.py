import pandas as pd
import glob, os

def analyze_dataframe_results(df, exp_name, cot=False):
    """
    Function to count valid and invalid responses from a DataFrame and return various statistics.

    Parameters:
    - df (pd.DataFrame): A DataFrame containing 'response' and 'model_name' columns.

    Returns:
    - dict: A dictionary containing the following statistics:
        - model_name (str): The model name (consistent across the DataFrame).
        - total_rows (int): The total number of rows (entries) in the DataFrame.
        - betray_count (int): The number of times "betray" appeared in the 'response' column.
        - cooperate_count (int): The number of times "cooperate" appeared in the 'response' column.
        - invalid_count (int): The number of invalid responses (neither "betray" nor "cooperate").
        - betray_percentage (float): The percentage of "betray" responses relative to total rows.
        - cooperate_percentage (float): The percentage of "cooperate" responses relative to total rows.
        - invalid_percentage (float): The percentage of invalid responses relative to total rows.
    """
    # Ensure the required columns exist
    if df is None or 'response' not in df.columns or 'model_name' not in df.columns:
        raise ValueError("The DataFrame must contain 'response' and 'model_name' columns.")

    # Lowercase the 'response' column
    df['response_lower'] = df['response'].str.lower().str.strip()

    # Get the model name (assuming it's consistent across the file)
    model_name = df['model_name'].iloc[0]  # Take the first value as the model name
    prisoner = df['prisoner'].iloc[0]

    if not cot:
        # Check if each response is valid ('betray' or 'cooperate')
        valid_responses = df['response_lower'].isin(['betray', 'cooperate'])
        
        # Total number of rows (runs)
        total_rows = len(df)
        
        # Count occurrences of "betray" and "cooperate"
        betray_count = (df['response_lower'] == 'betray').sum()
        cooperate_count = (df['response_lower'] == 'cooperate').sum()
        
        # Count invalid responses (neither "betray" nor "cooperate")
        invalid_count = (~valid_responses).sum()
    else:       
        # Total number of rows (runs)
        total_rows = len(df)
        
        betray_count = 0
        cooperate_count = 0
        invalid_count = 0
        for i in range(total_rows):
            if "<ans>BETRAY</ans>" in df['response'][i]:
                betray_count += 1
            elif "<ans>COOPERATE</ans>" in df['response'][i]:
                cooperate_count += 1
            else:
                invalid_count += 1               

    # Calculate percentages
    betray_percentage = (betray_count / total_rows) * 100 if total_rows > 0 else 0
    cooperate_percentage = (cooperate_count / total_rows) * 100 if total_rows > 0 else 0
    invalid_percentage = (invalid_count / total_rows) * 100 if total_rows > 0 else 0
	
    # Return the result as a dictionary
    return {
        'model_name': model_name,
        'exp_name': exp_name,
        'prisoner': prisoner,
        'total_rows': total_rows,
        'betray_count': betray_count,
        'cooperate_count': cooperate_count,
        'invalid_count': invalid_count,
        'betray_percentage': betray_percentage,
        'cooperate_percentage': cooperate_percentage,
        'invalid_percentage': invalid_percentage
	}

# Define a function to print the results in a neat format
def print_analysis_results(results, experiments):
    """
    Function to neatly print the analysis results for all models, organized by Prisoner A and Prisoner B.
    """
    # Get unique model names from the results
    # model_names = set(result['model_name'] for result in results)
    results_out = []
    for exp in experiments:
        print(exp)
        # Filter results for the current model and for Prisoner A and B
        prisoner_A_result = next(res for res in results if res['exp_name'] == exp and res['prisoner'] == 'A')
        prisoner_B_result = next(res for res in results if res['exp_name'] == exp and res['prisoner'] == 'B')
        
        print(f"Experiment: {exp}")
        # Print results for Prisoner A
        print(f"  Prisoner A:")
        print(f"    Total Runs: {prisoner_A_result['total_rows']}")
        print(f"    Betray Count: {prisoner_A_result['betray_count']} ({prisoner_A_result['betray_percentage']:.2f}%)")
        print(f"    Cooperate Count: {prisoner_A_result['cooperate_count']} ({prisoner_A_result['cooperate_percentage']:.2f}%)")
        print(f"    Invalid Count: {prisoner_A_result['invalid_count']} ({prisoner_A_result['invalid_percentage']:.2f}%)")
        
        # Print results for Prisoner B
        print(f"  Prisoner B:")
        print(f"    Total Runs: {prisoner_B_result['total_rows']}")
        print(f"    Betray Count: {prisoner_B_result['betray_count']} ({prisoner_B_result['betray_percentage']:.2f}%)")
        print(f"    Cooperate Count: {prisoner_B_result['cooperate_count']} ({prisoner_B_result['cooperate_percentage']:.2f}%)")
        print(f"    Invalid Count: {prisoner_B_result['invalid_count']} ({prisoner_B_result['invalid_percentage']:.2f}%)")

        # print(f"    Total Runs: {result['total_rows']}")
        # print(f"    Betray Count: {result['betray_count']} ({result['betray_percentage']:.2f}%)")
        # print(f"    Cooperate Count: {result['cooperate_count']} ({result['cooperate_percentage']:.2f}%)")
        # print(f"    Invalid Count: {result['invalid_count']} ({result['invalid_percentage']:.2f}%)")

        print("-" * 40 + '\n')
    #     result = {
    #         'exp_name': exp,
    #         'betray_percentage': (prisoner_B_result['betray_percentage']+prisoner_A_result['betray_percentage'])/2,
    #         'cooperate_percentage': (prisoner_B_result['cooperate_percentage']+prisoner_A_result['cooperate_percentage'])/2,
    #     }
    #     results_out.append(result)
    # pd.DataFrame(results_out).to_csv("results.csv")



def main():
    dir = "data/logs/repe-mistral-nemo/"

    experiments = []
    results = []
    for file in glob.glob(os.path.join(dir, "*")):
        df = pd.read_csv(file)
        exp_name = os.path.basename(file).strip(".csv").split("_")[0]
        if "test" not in file:
            if "cot" in file:
                res = analyze_dataframe_results(df, exp_name=exp_name, cot=True)
            else:
                res = analyze_dataframe_results(df, exp_name=exp_name)
            results.append(res)
            experiments.append(exp_name)

    # print_analysis_results(results, experiments)
    pd.DataFrame(results).to_csv("results.csv")

if __name__ == "__main__":
    main()