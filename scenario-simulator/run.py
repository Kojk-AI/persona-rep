import os

from helper.llm_helper import LLMHelper
llm_helper = LLMHelper()
from scenario_prompts import *
from tqdm.auto import tqdm

def run_single_round_experiment(model_name, prompt_a, prompt_b, prisoner_A_filename, prisoner_B_filename, llm_params={'temperature': 0}):
	# Run 100 iteration for prisoner A
	for _ in tqdm(range(10)):
		llm_helper.call_llm(
			system_prompt=PD_PRISONER_SYSTEM_PROMPT,
			prompt=prompt_a,
			model_name=model_name,
			llm_params=llm_params,
			prisoner='A',
			log_file=prisoner_A_filename
		)
	# Run 100 iteration for prisoner B
	for _ in tqdm(range(10)):
		llm_helper.call_llm(
			system_prompt=PD_PRISONER_SYSTEM_PROMPT,
			prompt=prompt_b,
			model_name=model_name,
			llm_params=llm_params,
			prisoner='B',
			log_file=prisoner_B_filename
		)
		
def run_test(model_name, prompt, log_filename, llm_params={'temperature': 0}):
	# Run 100 iteration for prisoner A
	for _ in range(1):
		llm_helper.call_llm(
			system_prompt=PD_PRISONER_SYSTEM_PROMPT,
			prompt=prompt,
			model_name=model_name,
			llm_params=llm_params,
			prisoner='test',
			log_file=log_filename
		)
		
def test_prompts(model_name, exp_name):
    test = [PD_PRISONER_A_USER_PROMPT_TEST_1,
            PD_PRISONER_A_USER_PROMPT_TEST_2,
            PD_PRISONER_A_USER_PROMPT_TEST_3,
            PD_PRISONER_A_USER_PROMPT_TEST_4,
            PD_PRISONER_A_USER_PROMPT_TEST_5,
            PD_PRISONER_A_USER_PROMPT_TEST_6,
            PD_PRISONER_A_USER_PROMPT_TEST_7,
            PD_PRISONER_A_USER_PROMPT_TEST_8,
            ]

    model_name = model_name
    test_filename = f'{model_name}_{exp_name}_test_result.csv'

    llm_params = {
        'temperature': 0.0,
        'coeff': 0.0,
        'direction': -1,
        'personality': 'openness',
    }

    for prompt in test:
        run_test(
            model_name=model_name,
            prompt=prompt,
            log_filename=test_filename,
            llm_params=llm_params
        )

def main(model_name):
    model_name = model_name
	
    for cot in tqdm([True, False]):
        exp_name = 'baseline'
        llm_params = {
            'temperature': 0.3,
            'coeff': 0.0,
            'direction': -1,
            'personality': 'openness',
        }

        if cot:
            prisoner_A_filename = f'{model_name}-cot-{exp_name}_prisoner_A_result.csv'
            prisoner_B_filename = f'{model_name}-cot-{exp_name}_prisoner_B_result.csv'
            prompt_a = PD_PRISONER_A_USER_PROMPT_COT
            prompt_b = PD_PRISONER_B_USER_PROMPT_COT
        else:
            prisoner_A_filename = f'{model_name}-{exp_name}_prisoner_A_result.csv'
            prisoner_B_filename = f'{model_name}-{exp_name}_prisoner_B_result.csv'
            prompt_a = PD_PRISONER_A_USER_PROMPT
            prompt_b = PD_PRISONER_B_USER_PROMPT

        test_prompts(model_name=model_name, exp_name=exp_name)

        run_single_round_experiment(
            model_name=model_name,
            prompt_a=prompt_a,
            prompt_b=prompt_b,
            prisoner_A_filename=prisoner_A_filename,
            prisoner_B_filename=prisoner_B_filename,
            llm_params=llm_params,
        )

        for personality in tqdm(["extraversion", "agreeableness", "openness", "conscientiousness", "neuroticism"]):
            for direction in tqdm(["plus", "minus"]):
                #steering
                if direction == "plus":
                    llm_params = {
                        'temperature': 0.3,
                        'coeff': 3.5,
                        'direction': 1,
                        'personality': personality,
                    }
                else:
                    llm_params = {
                        'temperature': 0.3,
                        'coeff': 3.5,
                        'direction': -1,
                        'personality': personality,
                    }
                exp_name = personality + "-" + direction
                prisoner_A_filename = f'{model_name}-{exp_name}_prisoner_A_result.csv'
                prisoner_B_filename = f'{model_name}-{exp_name}_prisoner_B_result.csv'

                run_single_round_experiment(
                    model_name=model_name,
                    prompt_a=prompt_a,
                    prompt_b=prompt_b,
                    prisoner_A_filename=prisoner_A_filename,
                    prisoner_B_filename=prisoner_B_filename,
                    llm_params=llm_params,
                )

                test_prompts(model_name=model_name, exp_name=exp_name)

                #prompting
                if direction == "plus":
                    llm_params = {
                        'temperature': 0.3,
                        'coeff': 0.0,
                        'direction': 1,
                        'personality': personality,
                    }           
                    prompt_a_prompt = f"Your personality is 100% {personality} based on the big 5 personality traits. " + prompt_a
                    prompt_b_prompt = f"Your personality is 100% {personality} based on the big 5 personality traits. " + prompt_b
                else:
                    llm_params = {
                        'temperature': 0.3,
                        'coeff': 0.0,
                        'direction': -1,
                        'personality': personality,
                    }
                    prompt_a_prompt = f"Your personality is 0% {personality} based on the big 5 personality traits. " + prompt_a
                    prompt_b_prompt = f"Your personality is 0% {personality} based on the big 5 personality traits. " + prompt_b

                exp_name = personality + "-" + direction + "-prompting"
                prisoner_A_filename = f'{model_name}-{exp_name}_prisoner_A_result.csv'
                prisoner_B_filename = f'{model_name}-{exp_name}_prisoner_B_result.csv'

                run_single_round_experiment(
                    model_name=model_name,
                    prompt_a=prompt_a_prompt,
                    prompt_b=prompt_b_prompt,
                    prisoner_A_filename=prisoner_A_filename,
                    prisoner_B_filename=prisoner_B_filename,
                    llm_params=llm_params,
                )

                test_prompts(model_name=model_name, exp_name=exp_name)

if __name__ == "__main__":
    model_name = "repe-mistral-nemo"
    main(model_name=model_name)