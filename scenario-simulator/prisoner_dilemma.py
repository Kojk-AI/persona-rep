import random
from typing import Callable, Optional, Tuple
from enum import Enum
from helper.llm_helper import LLMHelper  # Correctly importing LLMHelper

# from scenario_prompts import *

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
        llm_params: dict,
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
        self.llm_params = llm_params
        self.player2_strategy = player2_strategy
        self.n_rounds = n_rounds
        self.n_iterations = n_iterations
        self.seed = seed
        self.prisoner_A_filename = prisoner_A_filename
        self.log_test_filename = prisoner_A_filename.strip(".txt") + "_test.txt"
        self.result_filename = prisoner_A_filename.strip(".txt") + "_history.json"
        random.seed(seed)

        # Instantiate the LLMHelper class for calling the LLM
        self.llm_helper = LLMHelper()

    def simulate(self, cot: bool):
        """
        Simulate the game for the specified number of iterations and rounds.
        """
        for iteration in range(self.n_iterations):
            history = []  # To store the history of each round's actions
            defected = False  # Track defection for strategies like GRIM_TRIGGER

            for round_num in range(self.n_rounds):
                # Create history summary for Player 1 (LLM)
                history_summary = self._create_summary_string(history, round_num)

                # Call the LLM to get Player 1's action
                player1_action = self._player1_fn_using_llm(
                    cot, history_summary, round_num
                )

                # Get Player 2's action based on its strategy
                player2_action, defected = self._get_player2_action(
                    defected, player1_action, round_num, history
                )

                # Update history with actions for this round
                history.append((player1_action, player2_action))

                # Print or log the actions (Optional)
                print(
                    f"Round {round_num + 1}: Player 1 (LLM) = {player1_action}, Player 2 (Strategy) = {player2_action}"
                )
            with open(self.result_filename, "w") as f:
                json.dump(history, f)

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

    def _player1_fn_using_llm(
        self, cot: bool, history_summary: str, round_num: int
    ) -> str:
        """
        Simulates Player 1's action by calling the LLM.

        Args:
            history_summary (str): The summary of previous rounds to pass to the LLM.
            round_num (int): The current round number.

        Returns:
            str: Player 1's action ('Cooperate' or 'Defect').
        """
        PD_PRISONER_SYSTEM_PROMPT = """
        You are playing an Iterated Prisoner's Dilemma game. In each round, you must choose whether to cooperate or defect.

        The rules are as follows:
        - If both prisoners cooperate, both prisoners will each serve 1 year in prison.
        - If both prisoners defect, both prisoners will each serve 3 years in prison.
        - If prisoner A cooperates and prisoner B defects, prisoner A will serve 5 years in prison, and prisoner B will go free (0 years).
        - If prisoner A detects and prisoner B cooperates, prisoner A will go free (0 years), and prisoner B will serve 5 years.

        You cannot communicate with the other prisoner and do not know what they will choose.

        The game is played repeatedly against another player. You will be provided with a summary of all previous rounds, showing the actions taken by both you and your opponent. Based on this information, you need to decide whether to cooperate or defect in the current round.

        You are playing as Prisoner A.
        """

        if cot:
            PD_PRISONER_USER_PROMPT = f"""
            The following is a summary of the previous rounds:
            {history_summary}
            What is your action for this round? First, think step by step and then reply with either <ans>cooperate</ans> or <ans>defect</ans>.
            """
        else:
            PD_PRISONER_USER_PROMPT = f"""
            The following is a summary of the previous rounds:
            {history_summary}
            What is your action for this round? Please respond only with one word: either 'cooperate' or 'defect'.
            """

        # test_prompts = [
        #     PD_PRISONER_A_USER_PROMPT_TEST_1,
        #     PD_PRISONER_A_USER_PROMPT_TEST_2,
        #     PD_PRISONER_A_USER_PROMPT_TEST_3,
        #     PD_PRISONER_A_USER_PROMPT_TEST_4,
        #     PD_PRISONER_A_USER_PROMPT_TEST_5,
        #     PD_PRISONER_A_USER_PROMPT_TEST_6,
        #     PD_PRISONER_A_USER_PROMPT_TEST_7,
        #     PD_PRISONER_A_USER_PROMPT_TEST_8,
        #     PD_PRISONER_A_USER_PROMPT_TEST_9,
        #     PD_PRISONER_A_USER_PROMPT_TEST_10,
        #     PD_PRISONER_A_USER_PROMPT_TEST_11,
        #     PD_PRISONER_A_USER_PROMPT_TEST_12,
        #     PD_PRISONER_A_USER_PROMPT_TEST_13,
        # ]

        # for i, test in enumerate(test_prompts):

        #     if i == 9 or i == 10:
        #         r = random.randint(1, i - 1)
        #         test = test.format(r)

        #     prompt = f"""
        #     The following is a summary of the previous rounds:
        #     {history_summary}
        #     {test}
        #     """
        #     self.llm_helper.call_llm(
        #         system_prompt=PD_PRISONER_SYSTEM_PROMPT,
        #         prompt=prompt,
        #         model_name=self.model_name,
        #         llm_params=self.llm_params,
        #         prisoner="test",
        #         log_file=self.log_test_filename,
        #     )

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
        if cot:
            if "<ans>cooperate</ans>" in response[self.model_name][0]["response"]:
                return "cooperate"
            elif "<ans>defect</ans>" in response[self.model_name][0]["response"]:
                return "defect"
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
