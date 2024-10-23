import random
from typing import Optional, Dict, Callable, List, Tuple
from enum import Enum


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
    A class to simulate the Iterated Prisoner's Dilemma with two players.
    Player 1 is controlled externally via a function call to LLM, and Player 2 follows a defined strategy.

    Attributes:
        n_rounds (int): The number of rounds to simulate in a single experiment.
        n_iterations (int): The number of times the experiment is repeated.
        player2_strategy (PlayerStrategy): The strategy for Player 2 (defined by an Enum).
        player1_fn (Callable[[str], str]): A function controlling Player 1's behavior, which takes the opponent's last action and returns Player 1's action.
        metrics (List[Dict[str, float]]): Stores metrics for Player 2 for each iteration.
    """

    def __init__(
        self,
        player1_fn: Callable[[str], str],
        player2_strategy: PlayerStrategy,
        n_rounds: int = 10,
        n_iterations: int = 1,
        seed: int = 42,
    ):
        """
        Initializes the Iterated Prisoner's Dilemma simulation.

        Args:
            player1_fn (Callable[[str], str]): A function that returns Player 1's action based on the opponent's last action.
            player2_strategy (PlayerStrategy): The strategy for Player 2 (e.g., TIT_FOR_TAT, GRIM_TRIGGER, etc.).
            n_rounds (int): The number of rounds to simulate in a single experiment.
            n_iterations (int): The number of times the entire experiment will be repeated.
            seed (int): Random seed for reproducibility in random strategies.
        """
        self.n_rounds = n_rounds
        self.n_iterations = n_iterations
        self.player2_strategy = player2_strategy
        self.player1_fn = player1_fn
        self.seed = seed
        random.seed(seed)
        self.metrics = []  # Store metrics for each iteration

    def simulate(self) -> List[Dict[str, float]]:
        """
        Runs the simulation for the specified number of iterations.
        Also, stores the history of each round's actions.

        Returns:
            List[Dict[str, float]]: A list of final metrics for Player 2 from each iteration.
        """
        histories = []
        for iteration in range(self.n_iterations):
            random.seed(self.seed + iteration)  # Reset random seed for each iteration
            experiment_metrics, history = (
                self._simulate_single_experiment()
            )  # Get both metrics and history
            self.metrics.append(experiment_metrics)
            histories.append(history)
        self.histories = histories  # Store histories as an instance attribute
        return self.metrics

    def _simulate_single_experiment(
        self,
    ) -> Tuple[Dict[str, float], List[Tuple[str, str]]]:
        """
        Simulates a single experiment of multiple rounds of the Iterated Prisoner's Dilemma.

        Returns:
            Tuple[Dict[str, float], List[Tuple[str, str]]]: Final metrics and history for Player 2 after all rounds in this experiment.
        """
        previous_action = None  # Player 2's previous action
        defected = False  # Whether Player 1 has defected (for GRIM strategy)
        history_summary = ""  # Accumulates the summary of previous rounds for Player 1

        metrics_tracker = PrisonersDilemmaMetrics(
            self.n_rounds
        )  # Metrics for this experiment
        history = []  # Store the history of actions

        for round_num in range(self.n_rounds):
            # Create a history summary string based on previous rounds
            if round_num > 0:
                history_summary = self._create_summary_string(history, round_num)
                print("================")
                print("Round", round_num)
                print("================")
                print(history_summary)
                print("\n\n")

            # Get Player 1's action by passing the summary of all previous rounds
            player1_action = self.player1_fn(history_summary)

            # Player 2's action for this round using the given strategy.
            result = self._play_round(
                strategy=self.player2_strategy,
                opponent_action=player1_action,
                previous_action=previous_action,
                defected=defected,
                round_num=round_num,
            )
            player2_action = result["action"]
            defected = result["defected"]

            # Update history and metrics after each round
            history.append((player1_action, player2_action))
            metrics_tracker.update_metrics(player2_action, player1_action, round_num)

            # Update Player 2's previous action for strategies like WSLS
            previous_action = player2_action

        # Return the final metrics and the history of actions for this experiment
        return metrics_tracker.calculate_metrics(), history

    def _create_summary_string(
        self, history: List[Tuple[str, str]], round_num: int
    ) -> str:
        """
        Creates a summary string of all previous rounds to pass to Player 1.

        Args:
            history (List[Tuple[str, str]]): The history of actions for all previous rounds.
            round_num (int): The current round number.

        Returns:
            str: A summary string that provides Player 1 with information about all previous rounds.
        """
        summary = [
            f"Round {i+1}: Player1={p1}, Player2={p2}"
            for i, (p1, p2) in enumerate(history[:round_num])
        ]
        return "\n".join(summary)

    def _play_round(
        self,
        strategy: PlayerStrategy,
        opponent_action: Optional[str],
        previous_action: Optional[str],
        defected: bool,
        round_num: int,
    ) -> Dict[str, Optional[bool]]:
        """
        Determines the action for Player 2 based on the specified strategy.

        Args:
            strategy (PlayerStrategy): The strategy for Player 2.
            opponent_action (Optional[str]): The opponent's last action ("Cooperate" or "Defect").
            previous_action (Optional[str]): Player 2's previous action for WSLS.
            defected (bool): Whether the opponent defected earlier (used for GRIM strategy).
            round_num (int): The current round number.

        Returns:
            Dict[str, Optional[bool]]: The action for Player 2, whether the opponent has defected.
        """
        random.seed(self.seed + round_num)  # Ensure randomness varies each round

        # Initialize action with a default value
        action = "Cooperate"

        # Strategy logic
        if strategy == PlayerStrategy.ALWAYS_COOPERATE:
            action = "Cooperate"
        elif strategy == PlayerStrategy.ALWAYS_DEFECT:
            action = "Defect"
        elif strategy == PlayerStrategy.RANDOM:
            action = random.choice(["Cooperate", "Defect"])
        elif strategy == PlayerStrategy.UNFAIR_RANDOM:
            action = "Cooperate" if random.random() < 0.7 else "Defect"
        elif strategy == PlayerStrategy.TIT_FOR_TAT:
            action = "Cooperate" if round_num == 0 else opponent_action or "Cooperate"
        elif strategy == PlayerStrategy.SUSPICIOUS_TIT_FOR_TAT:
            action = "Defect" if round_num == 0 else opponent_action or "Cooperate"
        elif strategy == PlayerStrategy.GRIM_TRIGGER:
            if defected:
                action = "Defect"
            else:
                action = "Defect" if opponent_action == "Defect" else "Cooperate"
                if opponent_action == "Defect":
                    defected = True
        elif strategy == PlayerStrategy.WIN_STAY_LOSE_SHIFT:
            if round_num == 0:
                action = "Cooperate"
            else:
                action = (
                    previous_action
                    if opponent_action == previous_action
                    else ("Defect" if previous_action == "Cooperate" else "Cooperate")
                )

        return {"action": action, "defected": defected}


class PrisonersDilemmaMetrics:
    """
    Tracks and calculates the metrics for Player 2 during the game, including:
    Nice, Forgiving, Retaliatory, Troublemaking, and Emulative.
    """

    def __init__(self, n_rounds: int):
        """
        Initializes the metrics tracker for Player 2.

        Args:
            n_rounds (int): The number of rounds to simulate in a single experiment.
        """
        self.n_rounds = n_rounds
        self.opponent_defections = 0
        self.player_defections = 0
        self.first_defect = False
        self.penalties = 0
        self.forgiven_defection = 0
        self.reactions = 0
        self.uncalled_defections = 0
        self.occasions_to_provoke = 0
        self.mimics = 0
        self.history: List[Tuple[str, str]] = []

    def update_metrics(self, player_action: str, opponent_action: str, round_num: int):
        """
        Updates the metrics for Player 2 based on the actions of both players.

        Args:
            player_action (str): The action taken by Player 2.
            opponent_action (str): The action taken by Player 1.
            round_num (int): The current round number.
        """
        self.history.append((player_action, opponent_action))
        if not self.first_defect and player_action == "Defect":
            self.first_defect = True
        if player_action == "Defect":
            self.player_defections += 1
        if opponent_action == "Defect":
            self.opponent_defections += 1
        if opponent_action == "Defect" and round_num > 0:
            previous_round = self.history[round_num - 1]
            if previous_round[0] == "Cooperate" and player_action == "Cooperate":
                self.forgiven_defection += 1
            self.penalties += 1
        if opponent_action == "Defect" and player_action == "Defect":
            self.reactions += 1
        if round_num > 0:
            previous_round = self.history[round_num - 1]
            if previous_round[1] == "Cooperate" and player_action == "Defect":
                self.uncalled_defections += 1
            if previous_round[0] == "Cooperate" and previous_round[1] == "Cooperate":
                self.occasions_to_provoke += 1
        if round_num > 0:
            previous_round = self.history[round_num - 1]
            if player_action == previous_round[1]:
                self.mimics += 1

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculates and returns the final metrics for Player 2.

        Returns:
            Dict[str, float]: The calculated metrics.
        """
        return {
            "Nice": 1 if not self.first_defect else 0,
            "Forgiving": (
                self.forgiven_defection / self.penalties if self.penalties > 0 else 0
            ),
            "Retaliatory": (
                self.reactions / self.opponent_defections
                if self.opponent_defections > 0
                else 0
            ),
            "Troublemaking": (
                self.uncalled_defections / self.occasions_to_provoke
                if self.occasions_to_provoke > 0
                else 0
            ),
            "Emulative": (
                self.mimics / (self.n_rounds - 1) if self.n_rounds > 1 else 0
            ),
        }
