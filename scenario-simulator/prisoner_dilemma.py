import random
from typing import Callable, Optional, Tuple
from enum import Enum
from helper.llm_helper import LLMHelper  # Correctly importing LLMHelper
from scenario_prompts import *

import json

class PlayerStrategy(Enum):
    """
    Enum to represent the different strategies Player 2 can follow.
    """

    ALWAYS_COOPERATE = "AC"
    ALWAYS_DEFECT = "AD"
    RANDOM = "RND"
    UNFAIR_RANDOM = "URNDp"
    TIT_FOR_TAT = "TFT"
    SUSPICIOUS_TIT_FOR_TAT = "STFT"
    GRIM_TRIGGER = "GRIM"
    WIN_STAY_LOSE_SHIFT = "WSLS"


class PrisonersDilemma:
    """
    A class to simulate the Iterated Prisoner's Dilemma with Player 1 as the LLM
    and Player 2 following a predefined strategy.
    """

    def __init__(
        self,
        model_name: str,
        llm_parms: dict,
        llm_parms_B: dict,
        player2_strategy: PlayerStrategy,
        n_rounds: int = 10,
        n_iterations: int = 1,
        seed: int = 42,
        prisoner_A_filename: str = "prisoner_A_log.csv",
    ):
        """
        Initialize the game simulation.

        Args:
            model_name (str): The name of the LLM model used for Player 1.
            player2_strategy (PlayerStrategy): The strategy for Player 2.
            n_rounds (int): Number of rounds to simulate in each iteration.
            n_iterations (int): Number of times the entire experiment is repeated.
            seed (int): Random seed for reproducibility.
            prisoner_A_filename (str): Log file for Player 1 (LLM).
        """
        self.model_name = model_name
        self.llm_params = llm_parms
        self.llm_params_B = llm_parms_B
        self.player2_strategy = player2_strategy
        self.n_rounds = n_rounds
        self.n_iterations = n_iterations
        self.seed = seed
        self.prisoner_A_filename = prisoner_A_filename
        self.log_test_filename = prisoner_A_filename.strip(".txt") + "_test.txt"
        self.result_filename = prisoner_A_filename.strip(".txt") + "_history_iteration_{}.json"
        self.comms_filename = prisoner_A_filename.strip(".txt") + "_history_iteration_comms_{}.json"
        self.invalid_counts = 0
        random.seed(seed)

        # Instantiate the LLMHelper class for calling the LLM
        self.llm_helper = LLMHelper()

    def simulate(self, cot: bool, test: bool):
        """
        Simulate the game for the specified number of iterations and rounds.
        """
        for iteration in range(self.n_iterations):
            history = []  # To store the history of each round's actions
            defected = False  # Track defection for strategies like GRIM_TRIGGER

            if iteration>0:
                test = False

            player1_action = None
            
            for round_num in range(self.n_rounds):
                # Create history summary for Player 1 (LLM)
                history_summary = self._create_summary_string(history, round_num)

                # Call the LLM to get Player 1's action
                player1_action_new = self._player1_fn_using_llm(cot, history_summary, round_num, test)

                # Get Player 2's action based on its strategy
                player2_action, defected = self._get_player2_action(
                    defected, player1_action, round_num, history
                )

                player1_action = player1_action_new
                # Update history with actions for this round
                history.append((player1_action, player2_action))

                # Print or log the actions (Optional)
                print(
                    f"Round {round_num + 1}: Player 1 (LLM) = {player1_action}, Player 2 (Strategy) = {player2_action}"
                )
            with open(self.result_filename.format(iteration), "w") as f:
                json.dump(history, f)
                

    def simulate_comms(self, cot: bool, test: bool):
        """
        Simulate the game (with pre-play communication) for the specified number of iterations and rounds.
        """
        for iteration in range(self.n_iterations):
            history = []  # To store the history of each round's actions
            history_comms = []
            defected = False  # Track defection for strategies like GRIM_TRIGGER

            if iteration>0:
                test = False

            player1_action = None
            
            for round_num in range(self.n_rounds):
                #Get Player 2's comms
                player2_action_comms = self._get_player2_comms(
                    defected, player1_action, round_num, history
                )
                history_summary = self._create_summary_string_comms(history, history_comms, round_num, player2_action_comms)
                
                #Get Player 1's comms and action
                player1_action_comms, player1_action_new = self._player1_fn_using_llm_comms(cot, history_summary, round_num, test)
                
                history_comms.append((player1_action_comms, player2_action_comms))

                # Get Player 2's action
                player2_action, defected = self._get_player2_action_comms(
                    defected, player1_action, round_num, history, player1_action_comms
                )
                
                player1_action = player1_action_new
                history.append((player1_action, player2_action))

                print(
                    f"Round {round_num + 1}: Player 1 (LLM) = {player1_action}, Player 2 (Strategy) = {player2_action}"
                )

            with open(self.result_filename.format(iteration), "w") as f:
                json.dump(history, f)
            with open(self.comms_filename.format(iteration), "w") as f:
                json.dump(history_comms, f)

    def simulate_llmvllm_comms(self, cot: bool, test: bool):
        """
        Simulate the game (with pre-play communication and LLM for Player 2) for the specified number of iterations and rounds.
        """
        for iteration in range(self.n_iterations):
            history = []  # To store the history of each round's actions
            history_comms = []
            defected = False  # Track defection for strategies like GRIM_TRIGGER

            if iteration>0:
                test = False

            player1_action = None
            
            for round_num in range(self.n_rounds):
                history_summary = self._create_summary_string_comms(history, history_comms, round_num)

                player2_action_comms = self._player2_fn_using_llm_comms(
                    cot, history_summary, round_num, test
                )
                history_summary = self._create_summary_string_comms(history, history_comms, round_num, player2_action_comms)
                
                player1_action_comms, player1_action_new = self._player1_fn_using_llm_comms(cot, history_summary, round_num, test)
                
                history_comms.append((player1_action_comms, player2_action_comms))

                player2_action = self._player2_fn_using_llm_comms_action(
                    cot, history_summary, round_num, test
                )
                
                player1_action = player1_action_new
                history.append((player1_action, player2_action))

                print(
                    f"Round {round_num + 1}: Player 1 (LLM) = {player1_action}, Player 2 (Strategy) = {player2_action}"
                )

            with open(self.result_filename.format(iteration), "w") as f:
                json.dump(history, f)
            with open(self.comms_filename.format(iteration), "w") as f:
                json.dump(history_comms, f)
                
    def _create_summary_string_comms(self, history: list, history_comms: list, round_num: int, player2_comms = None) -> str:

        #Prefix with "No prior rounds for round num == 0"
        if round_num == 0:
            if len(history_comms) == 0:
                if player2_comms is None: 
                    out = "No prior rounds."
                else: 
                    out = f"No prior rounds. Prisoner B says that he will choose to {player2_comms} in the next round." 
            else:
                c1, c2 = history_comms[0]
                out = f"No prior rounds. You have told prisoner A that you will choose to {c2} in the next round. Prisoner A says that he will choose to {c1} in the next round."  
        else:
            out = []
            for i in range(len(history)):
                p1, p2 = history[i]
                c1, c2 = history_comms[i]
                out.append(f"Round {i + 1}: Prisoner A says that he will choose to {c1} and chooses to {p1}. Prisoner B says that he will choose to {c2} and chooses to {p2}.")
            if len(history_comms) != len(history):
                c1, c2 = history_comms[-1]
                out.append(f"You have told prisoner B that you will choose to {c1} in the next round. Prisoner B says that he will choose to {c2} in the next round.")
            elif player2_comms is not None:
                out.append(f"Prisoner B says that he will choose to {player2_comms} in the next round.")         
            
            out = "\n".join(out)
            
        return out
                        
    def _create_summary_string(self, history: list, round_num: int) -> str:
        """
        Generate a history summary string for Player 1 (LLM) based on previous rounds.

        Args:
            history (list): A list of tuples representing actions from previous rounds.
            round_num (int): The current round number.

        Returns:
            str: A formatted history summary string.
        """
        if round_num == 0:
            return "No prior rounds."
        return "\n".join(
            [
                f"Round {i + 1}: Prisoner A chooses to {p1}. Prisoner B chooses to {p2}."
                for i, (p1, p2) in enumerate(history)
            ]
        )

    def _player2_fn_using_llm_comms(self, cot: bool, history_summary: str, round_num: int, test: bool) -> str:

        PD_PRISONER_SYSTEM_PROMPT = """You are playing an Iterated Prisoner's Dilemma game. In each round, you must choose whether to cooperate or defect.
The rules are as follows:
- if both prisoners cooperate, both prisoners will each serve 1 year in prison.
- if both prisoners defect, both prisoners will each serve 3 years in prison.
- if prisoner A cooperates and prisoner B defects, prisoner A will serve 5 years in prison, and prisoner B will go free (0 years).
- if prisoner A detects and prisoner B cooperates, prisoner A will go free (0 years), and prisoner B will serve 5 years.
Serving less years in the prison is a good thing.
First, prisoner B will let you know his planned action, after which you will tell prisoner B what your planned action is. Prisoner B will be able to change his action depending on what you tell him. However, prisoner B will not inform if his action changes.
The game is played repeatedly against another player. You will be provided with a summary of all previous rounds, showing the actions taken by both you and your opponent and the communication between you two. Based on this information, you need to decide whether to cooperate or defect in the current round.
You are playing as Prisoner B."""

        if cot:
            PD_PRISONER_USER_PROMPT = f"""The following is a summary of the previous rounds:
{history_summary}
First, think step by step, using Theory of Mind and planning ahead, what will you tell Prisoner A about your action for the next round? Then respond with either <comms>cooperate</comms> or <comms>defect</comms> for what you will tell Prisoner A."""
        else:
            PD_PRISONER_USER_PROMPT = f"""The following is a summary of the previous rounds:
{history_summary}

What will you tell Prisoner A about your planned action for the next round? Please respond with either 'cooperate' or 'defect'.
            """
        
        if test:
            test_prompts = [PD_PRISONER_A_USER_PROMPT_TEST_1,
                PD_PRISONER_A_USER_PROMPT_TEST_2,
                PD_PRISONER_A_USER_PROMPT_TEST_3,
                PD_PRISONER_A_USER_PROMPT_TEST_4,
                PD_PRISONER_A_USER_PROMPT_TEST_5,
                PD_PRISONER_A_USER_PROMPT_TEST_6,
                PD_PRISONER_A_USER_PROMPT_TEST_7,
                PD_PRISONER_A_USER_PROMPT_TEST_8,
                PD_PRISONER_A_USER_PROMPT_TEST_9,
                PD_PRISONER_A_USER_PROMPT_TEST_10,
                PD_PRISONER_A_USER_PROMPT_TEST_11,
                PD_PRISONER_A_USER_PROMPT_TEST_12,
                PD_PRISONER_A_USER_PROMPT_TEST_13,
                ]
            
            for i, test in enumerate(test_prompts):
                
                if i==9 or i==10:
                    # r = random.randint(1,i-1)
                    for r in range(1,i-1,1):
                        test_ = test.format(r)
                
                        prompt = f"""
                        The following is a summary of the previous rounds:
                        {history_summary}
                        {test_}
                        """
                        self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=prompt,
                        model_name=self.model_name,
                        llm_params=self.llm_params_B,
                        prisoner='test',
                        log_file=self.log_test_filename
                        )
                else:
                        prompt = f"""
                        The following is a summary of the previous rounds:
                        {history_summary}
                        {test}
                        """
                        self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=prompt,
                        model_name=self.model_name,
                        llm_params=self.llm_params_B,
                        prisoner='test',
                        log_file=self.log_test_filename
                        )

        # Call the LLM using llm_helper
        response = self.llm_helper.call_llm(
            system_prompt=PD_PRISONER_SYSTEM_PROMPT,
            prompt=PD_PRISONER_USER_PROMPT,
            model_name=self.model_name,
            llm_params=self.llm_params_B,  # Deterministic behavior
            prisoner="A",
            round=round_num,
            log_file=self.prisoner_A_filename,
        )
        # print(response)
        # Extract action from LLM response
        action_invalid = False
        comms_invalid = True
        while action_invalid or comms_invalid:
            if cot:
                if "<comms>cooperate</comms>" in response[self.model_name][0]['response']:
                    comms_invalid = False
                    comms = "cooperate"
                elif "<comms>defect</comms>" in response[self.model_name][0]['response']:
                    comms_invalid = False
                    comms = "defect"
                if action_invalid or comms_invalid:
                    self.invalid_counts += 1
                    response = self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=PD_PRISONER_USER_PROMPT,
                        model_name=self.model_name,
                        llm_params=self.llm_params_B,  # Deterministic behavior
                        prisoner="A",
                        round=round_num,
                        log_file=self.prisoner_A_filename,
                    )
            else:
                if "cooperate" in response[self.model_name][0]["response"].lower():
                    return "cooperate"
                else:
                    return "defect"
        
        return comms

    def _player2_fn_using_llm_comms_action(self, cot: bool, history_summary: str, round_num: int, test: bool) -> str:

        PD_PRISONER_SYSTEM_PROMPT = """You are playing an Iterated Prisoner's Dilemma game. In each round, you must choose whether to cooperate or defect.
The rules are as follows:
- if both prisoners cooperate, both prisoners will each serve 1 year in prison.
- if both prisoners defect, both prisoners will each serve 3 years in prison.
- if prisoner A cooperates and prisoner B defects, prisoner A will serve 5 years in prison, and prisoner B will go free (0 years).
- if prisoner A detects and prisoner B cooperates, prisoner A will go free (0 years), and prisoner B will serve 5 years.
Serving less years in the prison is a good thing.
First, prisoner B will let you know his planned action, after which you will tell prisoner B what your planned action is. Prisoner B will be able to change his action depending on what you tell him. However, prisoner B will not inform if his action changes.
The game is played repeatedly against another player. You will be provided with a summary of all previous rounds, showing the actions taken by both you and your opponent and the communication between you two. Based on this information, you need to decide whether to cooperate or defect in the current round.
You are playing as Prisoner B."""

        if cot:
            PD_PRISONER_USER_PROMPT = f"""The following is a summary of the previous rounds:
{history_summary}
First, think step by step, using Theory of Mind and planning ahead, what is your actual action planned for the next round? Then respond with either <ans>cooperate</ans> or <ans>defect</ans> for your actual action."""
        else:
            PD_PRISONER_USER_PROMPT = f"""The following is a summary of the previous rounds:
{history_summary}

What will you tell Prisoner A about your planned action for the next round? Please respond with either 'cooperate' or 'defect'.
            """
        
        if test:
            test_prompts = [PD_PRISONER_A_USER_PROMPT_TEST_1,
                PD_PRISONER_A_USER_PROMPT_TEST_2,
                PD_PRISONER_A_USER_PROMPT_TEST_3,
                PD_PRISONER_A_USER_PROMPT_TEST_4,
                PD_PRISONER_A_USER_PROMPT_TEST_5,
                PD_PRISONER_A_USER_PROMPT_TEST_6,
                PD_PRISONER_A_USER_PROMPT_TEST_7,
                PD_PRISONER_A_USER_PROMPT_TEST_8,
                PD_PRISONER_A_USER_PROMPT_TEST_9,
                PD_PRISONER_A_USER_PROMPT_TEST_10,
                PD_PRISONER_A_USER_PROMPT_TEST_11,
                PD_PRISONER_A_USER_PROMPT_TEST_12,
                PD_PRISONER_A_USER_PROMPT_TEST_13,
                ]
            
            for i, test in enumerate(test_prompts):
                
                if i==9 or i==10:
                    # r = random.randint(1,i-1)
                    for r in range(1,i-1,1):
                        test_ = test.format(r)
                
                        prompt = f"""
                        The following is a summary of the previous rounds:
                        {history_summary}
                        {test_}
                        """
                        self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=prompt,
                        model_name=self.model_name,
                        llm_params=self.llm_params_B,
                        prisoner='test',
                        log_file=self.log_test_filename
                        )
                else:
                        prompt = f"""
                        The following is a summary of the previous rounds:
                        {history_summary}
                        {test}
                        """
                        self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=prompt,
                        model_name=self.model_name,
                        llm_params=self.llm_params_B,
                        prisoner='test',
                        log_file=self.log_test_filename
                        )

        # Call the LLM using llm_helper
        response = self.llm_helper.call_llm(
            system_prompt=PD_PRISONER_SYSTEM_PROMPT,
            prompt=PD_PRISONER_USER_PROMPT,
            model_name=self.model_name,
            llm_params=self.llm_params_B,  # Deterministic behavior
            prisoner="A",
            round=round_num,
            log_file=self.prisoner_A_filename,
        )
        # print(response)
        # Extract action from LLM response
        action_invalid = True
        comms_invalid = False
        while action_invalid or comms_invalid:
            if cot:
                if "<ans>cooperate</ans>" in response[self.model_name][0]['response']:
                    action_invalid = False
                    action = "cooperate"
                elif "<ans>defect</ans>" in response[self.model_name][0]['response']:
                    action_invalid = False
                    action = "defect"
                if action_invalid or comms_invalid:
                    self.invalid_counts += 1
                    response = self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=PD_PRISONER_USER_PROMPT,
                        model_name=self.model_name,
                        llm_params=self.llm_params_B,  # Deterministic behavior
                        prisoner="A",
                        round=round_num,
                        log_file=self.prisoner_A_filename,
                    )
            else:
                if "cooperate" in response[self.model_name][0]["response"].lower():
                    return "cooperate"
                else:
                    return "defect"
        
        return action
    
    def _player1_fn_using_llm_comms(self, cot: bool, history_summary: str, round_num: int, test: bool) -> str:

        PD_PRISONER_SYSTEM_PROMPT = """You are playing an Iterated Prisoner's Dilemma game. In each round, you must choose whether to cooperate or defect.
The rules are as follows:
- if both prisoners cooperate, both prisoners will each serve 1 year in prison.
- if both prisoners defect, both prisoners will each serve 3 years in prison.
- if prisoner A cooperates and prisoner B defects, prisoner A will serve 5 years in prison, and prisoner B will go free (0 years).
- if prisoner A detects and prisoner B cooperates, prisoner A will go free (0 years), and prisoner B will serve 5 years.
Serving less years in the prison is a good thing.
First, prisoner B will let you know his planned action, after which you will tell prisoner B what your planned action is. Prisoner B will be able to change his action depending on what you tell him. However, prisoner B will not inform if his action changes.
The game is played repeatedly against another player. You will be provided with a summary of all previous rounds, showing the actions taken by both you and your opponent and the communication between you two. Based on this information, you need to decide whether to cooperate or defect in the current round.
You are playing as Prisoner A."""

        if cot:
            PD_PRISONER_USER_PROMPT = f"""The following is a summary of the previous rounds:
{history_summary}
First, think step by step, using Theory of Mind and planning ahead, what will you tell Prisoner B about your action for the next round and what is your actual action planned for the next round? Respond with either <comms>cooperate</comms> or <comms>defect</comms> for what you will tell Prisoner B and respond with either <ans>cooperate</ans> or <ans>defect</ans> for your actual action."""
        else:
            PD_PRISONER_USER_PROMPT = f"""The following is a summary of the previous rounds:
{history_summary}

What will you tell Prisoner B about your planned action for the next round? Please respond with either 'cooperate' or 'defect'.
            """
        
        if test:
            test_prompts = [PD_PRISONER_A_USER_PROMPT_TEST_1,
                PD_PRISONER_A_USER_PROMPT_TEST_2,
                PD_PRISONER_A_USER_PROMPT_TEST_3,
                PD_PRISONER_A_USER_PROMPT_TEST_4,
                PD_PRISONER_A_USER_PROMPT_TEST_5,
                PD_PRISONER_A_USER_PROMPT_TEST_6,
                PD_PRISONER_A_USER_PROMPT_TEST_7,
                PD_PRISONER_A_USER_PROMPT_TEST_8,
                PD_PRISONER_A_USER_PROMPT_TEST_9,
                PD_PRISONER_A_USER_PROMPT_TEST_10,
                PD_PRISONER_A_USER_PROMPT_TEST_11,
                PD_PRISONER_A_USER_PROMPT_TEST_12,
                PD_PRISONER_A_USER_PROMPT_TEST_13,
                ]
            
            for i, test in enumerate(test_prompts):
                
                if i==9 or i==10:
                    # r = random.randint(1,i-1)
                    for r in range(1,i-1,1):
                        test_ = test.format(r)
                
                        prompt = f"""
                        The following is a summary of the previous rounds:
                        {history_summary}
                        {test_}
                        """
                        self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=prompt,
                        model_name=self.model_name,
                        llm_params=self.llm_params,
                        prisoner='test',
                        log_file=self.log_test_filename
                        )
                else:
                        prompt = f"""
                        The following is a summary of the previous rounds:
                        {history_summary}
                        {test}
                        """
                        self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=prompt,
                        model_name=self.model_name,
                        llm_params=self.llm_params,
                        prisoner='test',
                        log_file=self.log_test_filename
                        )

        # Call the LLM using llm_helper
        response = self.llm_helper.call_llm(
            system_prompt=PD_PRISONER_SYSTEM_PROMPT,
            prompt=PD_PRISONER_USER_PROMPT,
            model_name=self.model_name,
            llm_params=self.llm_params,  # Deterministic behavior
            prisoner="A",
            round=round_num,
            log_file=self.prisoner_A_filename,
        )
        # print(response)
        # Extract action from LLM response
        action_invalid = True
        comms_invalid = True
        while action_invalid or comms_invalid:
            if cot:
                if "<ans>cooperate</ans>" in response[self.model_name][0]['response']:
                    action_invalid = False
                    action = "cooperate"
                elif "<ans>defect</ans>" in response[self.model_name][0]['response']:
                    action_invalid = False
                    action = "defect"
                if "<comms>cooperate</comms>" in response[self.model_name][0]['response']:
                    comms_invalid = False
                    comms = "cooperate"
                elif "<comms>defect</comms>" in response[self.model_name][0]['response']:
                    comms_invalid = False
                    comms = "defect"
                if action_invalid or comms_invalid:
                    self.invalid_counts += 1
                    response = self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=PD_PRISONER_USER_PROMPT,
                        model_name=self.model_name,
                        llm_params=self.llm_params,  # Deterministic behavior
                        prisoner="A",
                        round=round_num,
                        log_file=self.prisoner_A_filename,
                    )
            else:
                if "cooperate" in response[self.model_name][0]["response"].lower():
                    return "cooperate"
                else:
                    return "defect"
        
        return comms, action
                
    def _player1_fn_using_llm(self, cot: bool, history_summary: str, round_num: int, test: bool) -> str:
        """
        Simulates Player 1's action by calling the LLM.

        Args:
            history_summary (str): The summary of previous rounds to pass to the LLM.
            round_num (int): The current round number.

        Returns:
            str: Player 1's action ('Cooperate' or 'Defect').
        """
        PD_PRISONER_SYSTEM_PROMPT = """You are playing an Iterated Prisoner's Dilemma game. In each round, you must choose whether to cooperate or defect.
The rules are as follows:
- if both prisoners cooperate, both prisoners will each serve 1 year in prison.
- if both prisoners defect, both prisoners will each serve 3 years in prison.
- if prisoner A cooperates and prisoner B defects, prisoner A will serve 5 years in prison, and prisoner B will go free (0 years).
- if prisoner A detects and prisoner B cooperates, prisoner A will go free (0 years), and prisoner B will serve 5 years.

Serving less years in the prison is a good thing.
First, prisoner B will let you know his planned action, after which you will tell prisoner B what your planned action is. Prisoner B will be able to change his action depending on what you tell him. However, prisoner B will not inform if his action changes.

The game is played repeatedly against another player. You will be provided with a summary of all previous rounds, showing the actions taken by both you and your opponent and the communication between you two. Based on this information, you need to decide whether to cooperate or defect in the current round.

You are playing as Prisoner A.
        """

        if cot:
            PD_PRISONER_USER_PROMPT = f"""The following is a summary of the previous rounds:
{history_summary}

First, think step by step, using Theory of Mind and planning ahead, what is your action for this round? Reply with either <ans>cooperate</ans> or <ans>defect</ans>.
            """
        else:
            PD_PRISONER_USER_PROMPT = f"""The following is a summary of the previous rounds:
{history_summary}

What is your action for this round? Please respond with either 'cooperate' or 'defect'.
            """
        
        if test:
            test_prompts = [PD_PRISONER_A_USER_PROMPT_TEST_1,
                PD_PRISONER_A_USER_PROMPT_TEST_2,
                PD_PRISONER_A_USER_PROMPT_TEST_3,
                PD_PRISONER_A_USER_PROMPT_TEST_4,
                PD_PRISONER_A_USER_PROMPT_TEST_5,
                PD_PRISONER_A_USER_PROMPT_TEST_6,
                PD_PRISONER_A_USER_PROMPT_TEST_7,
                PD_PRISONER_A_USER_PROMPT_TEST_8,
                PD_PRISONER_A_USER_PROMPT_TEST_9,
                PD_PRISONER_A_USER_PROMPT_TEST_10,
                PD_PRISONER_A_USER_PROMPT_TEST_11,
                PD_PRISONER_A_USER_PROMPT_TEST_12,
                PD_PRISONER_A_USER_PROMPT_TEST_13,
                ]
            
            for i, test in enumerate(test_prompts):
                
                if i==9 or i==10:
                    # r = random.randint(1,i-1)
                    for r in range(1,i-1,1):
                        test_ = test.format(r)
                
                        prompt = f"""
                        The following is a summary of the previous rounds:
                        {history_summary}
                        {test_}
                        """
                        self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=prompt,
                        model_name=self.model_name,
                        llm_params=self.llm_params,
                        prisoner='test',
                        log_file=self.log_test_filename
                        )
                else:
                        prompt = f"""
                        The following is a summary of the previous rounds:
                        {history_summary}
                        {test}
                        """
                        self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=prompt,
                        model_name=self.model_name,
                        llm_params=self.llm_params,
                        prisoner='test',
                        log_file=self.log_test_filename
                        )

        # Call the LLM using llm_helper
        response = self.llm_helper.call_llm(
            system_prompt=PD_PRISONER_SYSTEM_PROMPT,
            prompt=PD_PRISONER_USER_PROMPT,
            model_name=self.model_name,
            llm_params=self.llm_params,  # Deterministic behavior
            prisoner="A",
            round=round_num,
            log_file=self.prisoner_A_filename,
        )

        # print(response)
        # Extract action from LLM response
        invalid = True
        while invalid:
            if cot:
                if "<ans>cooperate</ans>" in response[self.model_name][0]['response']:
                    invalid = False
                    return "cooperate"
                elif "<ans>defect</ans>" in response[self.model_name][0]['response']:
                    invalid = False
                    return "defect"
                else:
                    self.invalid_counts += 1
                    response = self.llm_helper.call_llm(
                        system_prompt=PD_PRISONER_SYSTEM_PROMPT,
                        prompt=PD_PRISONER_USER_PROMPT,
                        model_name=self.model_name,
                        llm_params=self.llm_params,  # Deterministic behavior
                        prisoner="A",
                        round=round_num,
                        log_file=self.prisoner_A_filename,
                    )
            else:
                if "cooperate" in response[self.model_name][0]["response"].lower():
                    return "cooperate"
                else:
                    return "defect"

    def _get_player2_action(
        self, defected: bool, opponent_action: str, round_num: int, history: list
    ) -> Tuple[str, bool]:
        """
        Determines Player 2's action based on the selected strategy.

        Args:
            defected (bool): Whether Player 1 defected in a previous round (for GRIM_TRIGGER strategy).
            opponent_action (str): The action taken by Player 1 in the current round.
            round_num (int): The current round number.
            history (list): History of previous actions.

        Returns:
            Tuple[str, bool]: Player 2's action ('Cooperate' or 'Defect') and updated defected state.
        """
        if self.player2_strategy == PlayerStrategy.ALWAYS_COOPERATE:
            return "cooperate", defected
        elif self.player2_strategy == PlayerStrategy.ALWAYS_DEFECT:
            return "defect", defected
        elif self.player2_strategy == PlayerStrategy.RANDOM:
            return random.choice(["cooperate", "defect"]), defected
        elif self.player2_strategy == PlayerStrategy.UNFAIR_RANDOM:
            return "cooperate" if random.random() < 0.7 else "defect", defected
        elif self.player2_strategy == PlayerStrategy.TIT_FOR_TAT:
            return "cooperate" if round_num == 0 else opponent_action, defected
        elif self.player2_strategy == PlayerStrategy.SUSPICIOUS_TIT_FOR_TAT:
            return "defect" if round_num == 0 else opponent_action, defected
        elif self.player2_strategy == PlayerStrategy.GRIM_TRIGGER:
            if defected:
                return "defect", defected
            if opponent_action == "defect":
                return "defect", True  # Set defected to True
            return "cooperate", defected
        elif self.player2_strategy == PlayerStrategy.WIN_STAY_LOSE_SHIFT:
            if round_num == 0:
                return "cooperate", defected
            prev_p2_action, prev_p1_action = history[-1]
            if prev_p2_action == "cooperate" and prev_p1_action == "cooperate":
                return "cooperate", defected
            else:
                return "defect", defected

    def _get_player2_action_comms(
        self, defected: bool, opponent_action: str, round_num: int, history: list, player1_comms
    ) -> Tuple[str, bool]:
        """
        Determines Player 2's action based on the selected strategy.

        Args:
            defected (bool): Whether Player 1 defected in a previous round (for GRIM_TRIGGER strategy).
            opponent_action (str): The action taken by Player 1 in the current round.
            round_num (int): The current round number.
            history (list): History of previous actions.

        Returns:
            Tuple[str, bool]: Player 2's action ('Cooperate' or 'Defect') and updated defected state.
        """
        planned, _ = self._get_player2_action(defected, opponent_action, round_num, history)

        good_person = False
        if good_person:
            if planned == "defect" and player1_comms == "cooperate":
                return "cooperate", _
            elif planned == "cooperate" and player1_comms == "defect":
                return "defect", _
            else:
                return planned, _      
        else:  
            if planned == "cooperate" and player1_comms == "cooperate":
                return "defect", _
            elif planned == "cooperate" and player1_comms == "defect":
                return "defect", _
            else:
                return planned, _ 

    def _get_player2_comms(
        self, defected: bool, opponent_action: str, round_num: int, history: list
    ) -> Tuple[str, bool]:

        planned, _ = self._get_player2_action(defected, opponent_action, round_num, history)
        
        honest = True
        
        if honest:
            return planned
        else:
            if planned == "cooperate":
                return "defect"
            if planned == "defect":
                return "cooperate"