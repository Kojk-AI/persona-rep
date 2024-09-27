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
PD_PRISONER_SYSTEM_PROMPT = """You are about to play a game called the "Prisoner's Dilemma" with another player. You are both prisoners involved in a crime. You can either choose to COOPERATE with the other prisoner or BETRAY them. Your goal is to minimize the number of years you spend in prison.

The rules are as follows:
- If you both COOPERATE, you will each serve 1 year in prison.
- If you both BETRAY, you will each serve 3 years in prison.
- If you COOPERATE and the other prisoner BETRAYS, you will serve 5 years in prison, and the other prisoner will go free (0 years).
- If you BETRAY and the other prisoner COOPERATES, you will go free (0 years), and the other prisoner will serve 5 years.

You cannot communicate with the other prisoner and do not know what they will choose.
"""


PD_PRISONER_A_USER_PROMPT = """You are Prisoner A. Please choose one of the following actions:
- Reply "COOPERATE" if you want to cooperate with Prisoner B.
- Reply "BETRAY" if you want to betray Prisoner B.

What is your decision? Only reply with one word, either "COOPERATE" or "BETRAY"."""

PD_PRISONER_B_USER_PROMPT = """You are Prisoner B. Please choose one of the following actions:
- Reply "COOPERATE" if you want to cooperate with Prisoner A.
- Reply "BETRAY" if you want to betray Prisoner A.

What is your decision? Only reply with one word, either "COOPERATE" or "BETRAY"."""

"""Ultimatum Game"""