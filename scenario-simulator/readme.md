# Iterated Prisoner's Dilemma Simulation

This project simulates the **Iterated Prisoner's Dilemma**, a repeated game where two players independently choose to either cooperate or defect over a series of rounds. Player 1's actions are controlled by an external function, while Player 2 follows predefined strategies.

## Project Structure

```
├── prisoner_dillema.py       # Main simulation class and strategies for Player 2
├── scenario_prompts.py       # Contains scenario-specific prompts used for simulations
├── analysis/                 # Directory for analysis results or scripts
├── data/                     # Contains data files such as logs
│   └── logs/                 # Logs from simulations
│       └── archive/          # Archived logs
├── helper/                   # Helper scripts used by the main simulation
│   └── llm_helper/           # Facilitates LLM calls
│   └── custom_logger/        # Facilitates logging
├── tests/                    # Contains unit tests for validating the simulation
│   └── test_prisoners_dilemma.py          # Unit tests for the simulation
```

### Main Components:

1. **`prisoner_dillema.py`**: This file contains the main `PrisonersDilemma` class that implements the game logic. Player 2 can follow various strategies such as `Tit for Tat`, `Grim Trigger`, `Always Cooperate`, etc.

2. **`scenario_prompts.py`**: Stores specific scenario-based prompts that are used during simulations to interact with Player 1.

3. **`analysis/`**: A directory meant to store any scripts or results used to analyze the outcomes of the simulations.

4. **`data/logs/`**: Contains logs generated during the simulation runs. The `archive/` subfolder stores older logs.

5. **`helper/`**:

    - **`llm_helper/`**: Provides helper functions to facilitate communication with large language models (LLMs) for simulating Player 1’s responses.
    - **`custom_logger/`**: Contains logging functions to track and log activities during the simulation.

6. **`tests/`**:
    - **`test_prisoners_dilemma.py`**: Unit tests that verify the correctness of the strategies and functionality of the simulation. The tests validate Player 2’s behavior under various strategies.

## Setup Instructions

### Prerequisites

-   **Python 3.7+**
-   **Virtual environment**

### Installation

1. **Clone the repository**:

    ```bash
    git clone <your-repository-url>
    cd <your-project-directory>
    ```

2. **Create and activate a virtual environment**:

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. **Install any necessary dependencies**:
    ```bash
    pip install -r requirements.txt  # If a requirements file exists
    ```

### Running the Simulation

To run the simulation, customize the Player 2 strategy and Player 1’s behavior in `prisoner_dillema.py`. Here’s an example of how to set up the game and run it:

```python
from prisoner_dillema import PrisonersDilemma, PlayerStrategy

def player1_fn(history_summary):
    return "Cooperate"

game = PrisonersDilemma(
    player1_fn=player1_fn,
    player2_strategy=PlayerStrategy.TIT_FOR_TAT,
    n_rounds=10,
    n_iterations=1
)
results = game.simulate()
print(results)
```

### Running Unit Tests

Unit tests are located in the `tests/` directory and can be run with the following commands:

1. **With `unittest`**:

    ```bash
    python -m unittest discover -s tests
    ```

2. **With `pytest`** (if installed):
    ```bash
    pytest tests/
    ```

### Available Strategies for Player 2

-   **Always Cooperate (AC)**: Player 2 always cooperates.
-   **Always Defect (AD)**: Player 2 always defects.
-   **Tit for Tat (TFT)**: Player 2 mimics Player 1’s previous action.
-   **Suspicious Tit for Tat (STFT)**: Player 2 defects in the first round, then mimics Player 1.
-   **Grim Trigger (GRIM)**: Player 2 defects indefinitely after Player 1’s first defection.
-   **Win Stay Lose Shift (WSLS)**: Player 2 repeats its previous action if it matched Player 1’s, otherwise it switches.
-   **Random (RND)**: Player 2 randomly chooses to cooperate or defect.
-   **Unfair Random (URNDp)**: Player 2 cooperates more frequently than defecting (e.g., 70% cooperation).

### Logs and Analysis

Logs from simulations are stored in the `data/logs/` directory and can be used to analyze performance across rounds and strategies. The `analysis/` directory is available for storing any analysis scripts or results based on these logs.
