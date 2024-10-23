"""Prisoner's Dilemma"""

# PD_PRISONER_SYSTEM_PROMPT = """You are about to play a game called the "Prisoner's Dilemma" with another player. You are both prisoners involved in a crime. You can either choose to COOPERATE with the other prisoner or BETRAY them. Your goal is to minimize the number of years you spend in prison.

# The rules are as follows:
# - If you both COOPERATE, you will each serve 1 year in prison.
# - If you both BETRAY, you will each serve 3 years in prison.
# - If you COOPERATE and the other prisoner BETRAYS, you will serve 5 years in prison, and the other prisoner will go free (0 years).
# - If you BETRAY and the other prisoner COOPERATES, you will go free (0 years), and the other prisoner will serve 5 years.

# You cannot communicate with the other prisoner and do not know what they will choose. Your decision will be made independently.

# At the end of the round, I will inform you of the outcome based on the decisions made by you and the other prisoner.
# """
# PD_PRISONER_SYSTEM_PROMPT = """You are about to play a game called the "Prisoner's Dilemma" with another player. You are both prisoners involved in a crime. You can either choose to COOPERATE with the other prisoner or BETRAY them. Your goal is to minimize the number of years you spend in prison.

# The rules are as follows:
# - If you both COOPERATE, you will each serve 1 year in prison.
# - If you both BETRAY, you will each serve 3 years in prison.
# - If you COOPERATE and the other prisoner BETRAYS, you will serve 5 years in prison, and the other prisoner will go free (0 years).
# - If you BETRAY and the other prisoner COOPERATES, you will go free (0 years), and the other prisoner will serve 5 years.

# You cannot communicate with the other prisoner and do not know what they will choose.
# """


# PD_PRISONER_A_USER_PROMPT = """You are Prisoner A. Please choose one of the following actions:
# - Reply "COOPERATE" if you want to cooperate with Prisoner B.
# - Reply "BETRAY" if you want to betray Prisoner B.

# What is your decision? Only reply with one word, either "COOPERATE" or "BETRAY"."""

# PD_PRISONER_B_USER_PROMPT = """You are Prisoner B. Please choose one of the following actions:
# - Reply "COOPERATE" if you want to cooperate with Prisoner A.
# - Reply "BETRAY" if you want to betray Prisoner A.

# What is your decision? Only reply with one word, either "COOPERATE" or "BETRAY"."""


"""Updated"""

PD_PRISONER_SYSTEM_PROMPT = """You are playing an Iterated Prisoner's Dilemma game. In each round, you must choose whether to cooperate or defect.

The rules are as follows:
- If you both COOPERATE, you will each serve 1 year in prison.
- If you both BETRAY, you will each serve 3 years in prison.
- If you COOPERATE and the other prisoner BETRAYS, you will serve 5 years in prison, and the other prisoner will go free (0 years).
- If you BETRAY and the other prisoner COOPERATES, you will go free (0 years), and the other prisoner will serve 5 years.

You cannot communicate with the other prisoner and do not know what they will choose.

The game is played repeatedly against another player. You will be provided with a summary of all previous rounds, showing the actions taken by both you and your opponent. Based on this information, you need to decide whether to cooperate or defect in the current round.

You are playing as Player 1. After reviewing the history of previous rounds, respond only with one word: either 'Cooperate' or 'Defect' for the current round.
"""

PD_PRISONER_USER_PROMPT = """
The following is a summary of the previous rounds:

[history_summary]

What is your action for this round? Please respond with either 'Cooperate' or 'Defect'.
"""
PD_PRISONER_B_USER_PROMPT = """You are Prisoner B. Please choose one of the following actions:
- Reply "COOPERATE" if you want to cooperate with Prisoner A.
- Reply "BETRAY" if you want to betray Prisoner A.

What is your decision? Only reply with one word, either "COOPERATE" or "BETRAY"."""

PD_PRISONER_A_USER_PROMPT_COT = """You are Prisoner A. Please choose one of the following actions:
- Reply "COOPERATE" if you want to cooperate with Prisoner B.
- Reply "BETRAY" if you want to betray Prisoner B.

What is your decision? Think step by step and reply with either <ans>COOPERATE</ans> or <ans>BETRAY</ans>"."""

PD_PRISONER_B_USER_PROMPT_COT = """You are Prisoner B. Please choose one of the following actions:
- Reply "COOPERATE" if you want to cooperate with Prisoner A.
- Reply "BETRAY" if you want to betray Prisoner A.

What is your decision? Think step by step and reply with either <ans>COOPERATE</ans> or <ans>BETRAY</ans>"."""

PD_PRISONER_A_USER_PROMPT_TEST_1 = """What is the lowest/highest payoff player A can get in a single round?"""

PD_PRISONER_A_USER_PROMPT_TEST_2 = """Which actions is player A allowed to play?"""

PD_PRISONER_A_USER_PROMPT_TEST_3 = """What is the lowest/highest payoff player B can get in a single round?"""

PD_PRISONER_A_USER_PROMPT_TEST_4 = """Which actions is player B allowed to play?"""

PD_PRISONER_A_USER_PROMPT_TEST_5 = """Which is player X’s payoff in a single round if X betrays and Y betrays?"""

PD_PRISONER_A_USER_PROMPT_TEST_6 = """Which is player X’s payoff in a single round if X cooperates and Y cooperates?"""

PD_PRISONER_A_USER_PROMPT_TEST_7 = """Which is player X’s payoff in a single round if X betrays and Y cooperates?"""

PD_PRISONER_A_USER_PROMPT_TEST_8 = """Which is player X’s payoff in a single round if X cooperates and Y betrays?"""

"""Ultimatum Game"""
