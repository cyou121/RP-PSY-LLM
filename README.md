RP-PSY-LLM is a Python-based project designed to simulate and evaluate language model agents with specific BIG-5 personality traits through representation engineering.


## Setup
1. Install packages with pipenv.
   ```sh
    pipenv sync
    ```
2. Create a config file named `.env` which contains the following line:
    ```sh
    HUGGING_FACE_TOKEN=...
    ```
## Personality Simulation and Extraction
1. Personality Simulation.
    Run the following python code to perform personality simulation with representation engineering.
    ```sh
    python personality_simulation.py --n_adj 12 --n_trial 100 --output_dir RESULT_DIR
    ```
    This code creates a set of control vector for each BIG-5 personality dimension, which is then utilzed to simulate LLM agents with specific personality.
    For each LLM agent we perform the following task and save the results to the output directory specified:
    - personality test (IPIP)
    - text generation (pretend to be a human and talk about favorite activities)
2. Evaluation.
    - Perform analysis on IPIP test results (correlation test):
        ```sh
        python personality_eval.py --output_dir RESULT_DIR --ipip_analysis 
        ```
    - Perform analysis on text generation & perplexity:
        ```sh
        python personality_eval.py --output_dir RESULT_DIR --generation_analysis
        ```
