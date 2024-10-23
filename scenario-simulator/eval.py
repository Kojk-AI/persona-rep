import pandas as pd
import glob, os
from sfem import SFEM

def analyze_dataframe_results(df, df_opponent=None, exp_name="", cot=False):
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
	
    for d in [df,df_opponent]:
        if d is not None:
            d["betray"] = d.apply(lambda x:"<ans>BETRAY</ans>" in x['response'], axis=1)
            d["cooperate"] = d.apply(lambda x:"<ans>COOPERATE</ans>" in x['response'], axis=1)
            d["invalid"] = d.apply(lambda x:"<ans>COOPERATE</ans>" not in x['response'] and "<ans>BETRAY</ans>" not in x['response'], axis=1)
    
    #Reciprocity rate
    rr_count = 0
    rr_invalid = 0
    for row in range(1,df.shape[0]):
        if df['invalid'].iloc[row] == False and df_opponent['invalid'].iloc[row-1] == False:
            if df['betray'].iloc[row] == df_opponent['betray'].iloc[row-1]:
                rr_count +=1
        else:
            rr_invalid +=1
    rr_percentage = rr_count/(df.shape[0]-1)

    #Forgiveness rate based off paper "Nicer Than Humans: How do Large Language Models Behave in the Prisoner’s Dilemma?"
    f_count = 0
    opp_betray_count = 0
    penalty = 0
    f_invalid = 0
    vengeful = False
    for row in range(1,df.shape[0]):
        if df['invalid'].iloc[row] == False and df['invalid'].iloc[row-1] == False and df_opponent['invalid'].iloc[row-1] == False:
            if df_opponent['betray'].iloc[row-1] == True:
                opp_betray_count += 1
                if df['betray'].iloc[row] == False:
                    f_count += 1
                    vengeful = False
            if row > 1:
                if df_opponent['invalid'].iloc[row-2] == False:
                    if df_opponent['betray'].iloc[row-2] == True and df_opponent['betray'].iloc[row-1] == False:
                        if df['betray'].iloc[row] == True and df['betray'].iloc[row-1] == True:
                            penalty += 1
                            vengeful = True
                        else:
                            vengeful = False
                    else:
                        if vengeful and df['betray'].iloc[row] == True:
                            penalty += 1
                        else:
                            vengeful = False
                else:
                    f_invalid +=1
        else:
            f_invalid +=1

    if opp_betray_count == 0:
        f_percentage = -1
    else:
        f_percentage = f_count / (opp_betray_count + penalty)
    
    #Forgiveness rate based on  we only count forgiveness if the agent plays betrays after opponent betrays and then revert back to cooperation if the opponent seeks forgiveness (switches to cooperates). Penalty term remains.
    # f_count = 0
    # opp_betray_count = 0
    # penalty = 0
    # f_invalid = 0
    # vengeful = False
    # for row in range(1,df.shape[0]):
    #     if df['invalid'].iloc[row] == False and df_opponent['invalid'].iloc[row-1] == False:
    #         if df_opponent['betray'].iloc[row-1] == True:
    #             opp_betray_count += 1
    #         if row > 1:
    #             if df_opponent['betray'].iloc[row-2] == True and df_opponent['betray'].iloc[row-1] == False:
    #                 if df['betray'].iloc[row] == True and df['betray'].iloc[row-1] == True:
    #                     penalty += 1
    #                     vengeful = True
    #                 elif df['betray'].iloc[row] == True and df['betray'].iloc[row-1] == False:
    #                     f_count += 1
    #                     vengeful = False
    #         else:
    #             if vengeful and df['betray'].iloc[row] == True:
    #                 penalty += 1
    #             else:
    #                 vengeful = False
    #     else:
    #         f_invalid +=1

    # f_percentage = f_count / (opp_betray_count + penalty)

    #Exploitability
    e_count = 0
    opp_betray_count = 0
    e_invalid = 0
    for row in range(1,df.shape[0]):
        if df['invalid'].iloc[row] == False and df_opponent['invalid'].iloc[row-1] == False:
            if df_opponent['betray'].iloc[row-1] == True:
                opp_betray_count += 1
                if df['betray'].iloc[row] == False:
                    e_count += 1
        else:
            e_invalid +=1
            
    if opp_betray_count == 0:
        e_percentage = -1
    else:
        e_percentage = e_count / opp_betray_count

    #Retaliatory
    r_count = 0
    opp_betray_count = 0
    r_invalid = 0
    for row in range(1,df.shape[0]):
        if df['invalid'].iloc[row] == False and df['invalid'].iloc[row-1] and df_opponent['invalid'].iloc[row-1] == False:
            if df['betray'].iloc[row-1] == False and df_opponent['betray'].iloc[row-1] == True:
                opp_betray_count += 1
                if df['betray'].iloc[row] == True:
                    r_count += 1
        else:
            r_invalid +=1

    if opp_betray_count == 0:
        r_percentage = -1
    else:
        r_percentage = r_count / opp_betray_count

    #Troublemaking
    t_count = 0
    opportunity_count = 0
    t_invalid = 0
    for row in range(0,df.shape[0]):
        if row == 0:
            if df['invalid'].iloc[row] == False:
                opportunity_count += 1
                if df['betray'].iloc[row] == True:
                    t_count +=1
            else:
                t_invalid +=1 
        else:        
            if df['invalid'].iloc[row] == False and df_opponent['invalid'].iloc[row-1] == False:
                if df_opponent['betray'].iloc[row-1] == False:
                    opportunity_count += 1
                    if df['betray'].iloc[row] == True:
                        t_count += 1
            else:
                t_invalid +=1
                
    if opportunity_count == 0:
        t_percentage = -1
    else:
        t_percentage = t_count / opportunity_count

    #SFEM
    sfem = SFEM(df, df_opponent)
    sfem_res = sfem.save_results()
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
        'invalid_percentage': invalid_percentage,
        'reciprocity_rate': rr_percentage,
        'reciprocity_invalid': rr_invalid,
        'forgiveness_rate': f_percentage,
        'forgiveness_invalid': f_invalid,
        'exploitability_rate': e_percentage,
        'explotability_invalid': e_invalid,
        'retaliatory_rate': r_percentage,
        'rectaliatory_invalid': r_invalid,
        'troublemaking_rate': t_percentage,
        'troublemaking_invalid': t_invalid,    
        'sfem': sfem_res    
	}

# Define a function to print the results in a neat format
def print_analysis_results(results, experiments):
    """
    Function to neatly print the analysis results for all models, organized by Prisoner A and Prisoner B.
    """
    # Get unique model names from the results
    # model_names = set(result['model_name'] for result in results)
    results_out = []
    experiments = list(set(experiments))
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
        result = {
            'exp_name': exp,
            'betray_percentage': (prisoner_B_result['betray_percentage']+prisoner_A_result['betray_percentage'])/2,
            'cooperate_percentage': (prisoner_B_result['cooperate_percentage']+prisoner_A_result['cooperate_percentage'])/2,
        }
        results_out.append(result)
    pd.DataFrame(results_out).to_csv("results.csv")



def main():
    dir = "data/logs/repe-mistral-nemo-100/"

    experiments = []
    results = []
    for file in glob.glob(os.path.join(dir, "*")):
        if "test" not in file:
            df = pd.read_csv(file)
            exp_name = os.path.basename(file).strip(".csv").split("_")[0]
            if "test" not in file:
                if "cot" in file:
                    res = analyze_dataframe_results(df, exp_name=exp_name, cot=False)
                else:
                    res = analyze_dataframe_results(df, exp_name=exp_name, cot=True)
                results.append(res)
                experiments.append(exp_name)

    print_analysis_results(results, experiments)
    # pd.DataFrame(results).to_csv("results-temp.csv")

if __name__ == "__main__":
    main()