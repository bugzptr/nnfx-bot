Phase 1: Parameter Optimization Framework (Backtesting-based)
The goal here is to systematically test different combinations of strategy parameters to find sets that perform well on historical data.
Refactor Configuration Handling for Optimization:
Current State: StrategyConfig loads a single strategy_config.json.
Change:
Modify StrategyConfig (or create a new mechanism) to accept parameters directly as a dictionary, not just from a file. This will allow an optimization loop to pass different parameter sets.
The strategy_config_global instance might need to be managed differently if parameters are changing per backtest run, especially with multiprocessing. It might be better to pass parameter dictionaries directly to worker processes.
The _backtest_worker_process will need to accept a strategy_params_dict instead of a strategy_config_path_str and instantiate StrategyConfig using that dict.
Define Parameter Ranges and Steps:
Identify which parameters in your strategy_config.json you want to optimize (e.g., tema_period, cci_period, cmf_window, ATR multipliers, etc.).
For each parameter, define a realistic range and a step size for iteration (e.g., tema_period: from 10 to 50, step 5).
Store these optimization ranges in a separate configuration file or directly in the script for now.
Choose an Optimization Library/Method:
Grid Search: Simplest to implement. Tests every possible combination of parameters within the defined ranges. Can be computationally very expensive if many parameters or wide ranges.
Random Search: Randomly samples parameter combinations. Can be more efficient than grid search for high-dimensional spaces.
Bayesian Optimization (e.g., optuna, scikit-optimize/skopt): More advanced. Intelligently chooses the next set of parameters to test based on previous results, often converging faster to good solutions. optuna is highly recommended for its ease of use and powerful features.
Implement the Optimization Loop:
The loop will generate sets of parameters based on the chosen method (grid, random, or Optuna's suggestions).
For each parameter set:
Run a backtest (using DualNNFXSystem.scan_pairs or a modified version that backtests a specific parameter set across relevant symbols). You might focus on a smaller, representative set of symbols for optimization to save time, or optimize per symbol.
Collect a key performance metric to optimize (e.g., Sharpe ratio, profit factor, net profit, Calmar ratio, or your custom score).
The optimization library (like Optuna) will typically handle the process of suggesting new trials and keeping track of the best results.
Objective Function for Optuna (if used):
You'll define an "objective function" that takes a trial object (from Optuna).
Inside this function, you'll:
Use trial.suggest_int(), trial.suggest_float(), trial.suggest_categorical() to define the parameters and their ranges for Optuna to explore.
Create a strategy parameter dictionary from these suggestions.
Run the backtest(s) with these parameters.
Return the performance metric you want to maximize or minimize.
Example Optuna objective:
# import optuna
# def objective(trial, symbols_for_opt, api_config_dict):
#     params = {
#         "indicators": {
#             "tema_period": trial.suggest_int("tema_period", 10, 50, step=5),
#             "cmf_window": trial.suggest_int("cmf_window", 10, 30, step=2),
#             # ... other parameters ...
#         },
#         "risk_per_trade": trial.suggest_float("risk_per_trade", 0.01, 0.03, step=0.005)
#         # ... other sections ...
#     }
#     # Adapt StrategyConfig to take params dict or modify worker to use this dict
#     # For now, let's assume worker can take a strategy_params_dict
#     temp_strategy_config = StrategyConfig(config_path_str=None) # Dummy path
#     temp_strategy_config.params = params # Override params
#
#     # Run scan_pairs with these params for the selected symbols_for_opt
#     # This needs careful refactoring of how config is passed to workers
#     # For simplicity, assume a function `run_backtest_with_params` exists
#     # overall_performance_metric = run_backtest_with_params(api_config_dict, params, symbols_for_opt)
#     # return overall_performance_metric # e.g., average Sharpe across symbols_for_opt

# study = optuna.create_study(direction="maximize")
# study.optimize(lambda trial: objective(trial, ['BTCUSDT', 'ETHUSDT'], api_cfg_main), n_trials=100)
# logger.info(f"Best trial: {study.best_trial.params}, Value: {study.best_value}")
Use code with caution.
Python
Storing and Analyzing Optimization Results:
Optimization runs can generate a lot of data. Optuna stores results in a database (SQLite by default), allowing for easy analysis and resumption of studies.
Identify the best parameter sets based on your chosen metric.
Robustness Checks for Optimized Parameters:
Parameters optimized on one period might not perform well on another (overfitting). This leads to the need for forward testing.
Phase 2: Forward Testing (Walk-Forward Analysis)
Forward testing (or walk-forward analysis) is crucial to assess how well parameters optimized on past data perform on unseen future data, giving a more realistic expectation of live performance.
Data Splitting:
Divide your total historical data into multiple contiguous segments:
In-Sample (IS) / Training Sets: Used for parameter optimization.
Out-of-Sample (OOS) / Validation Sets: Used for forward testing the parameters optimized on the immediately preceding In-Sample set. These OOS sets must be "future" data relative to their IS set.
Walk-Forward Loop:
Iterate through your data segments:
Step A (Optimization): Take the current In-Sample period. Run your parameter optimization (from Phase 1) on this IS data to find the best parameter set (P_optimal_IS).
Step B (Forward Test): Take the next Out-of-Sample period (immediately following the IS period). Run a standard backtest on this OOS data using only the P_optimal_IS found in Step A. Do not re-optimize on the OOS data.
Step C (Record): Record the performance of P_optimal_IS on the OOS period.
Step D (Slide Windows): Move the In-Sample and Out-of-Sample windows forward in time and repeat from Step A. The IS window can be expanding or sliding.
Aggregate Forward Test Results:
Combine the performance metrics from all the OOS periods. This aggregated OOS performance gives a more reliable estimate of the strategy's viability than a single backtest on the entire dataset.
Look for consistency in performance across different OOS periods.
Implementation Details:
You'll need functions to slice your k-line data (Pandas DataFrames) based on date ranges for IS and OOS periods.
The main optimization loop from Phase 1 will be called repeatedly within the walk-forward loop for each IS period.
Carefully manage the passing of optimized parameters from the IS optimization step to the OOS backtesting step.
Phase 3: Live Paper Trading / Small Scale Live Trading (Optional, but Recommended before large capital)
Real-time Signal Generation: Your get_current_signals is a good start.
Broker Integration for Orders (if going live): This is a significant step, requiring integration with Bitget's trading APIs for placing/managing orders, checking balances, etc. Your current BitgetAPI class focuses on market data; it would need expansion.
Position Sizing and Risk Management in Real-Time.
Monitoring and Logging for Live Operations.
Paper Trading: Use the signals generated on live data but simulate trades without real capital.
Small Scale Live Trading: If paper trading is successful, trade with a small, controlled amount of capital to experience real-world execution, slippage, and API latencies.
Plan Outline:
Sprint 1: Basic Parameterization & Optuna Setup
* Task 1.1: Refactor StrategyConfig and _backtest_worker_process to accept strategy parameters as a dictionary.
* Task 1.2: Define optimization ranges for 2-3 key parameters (e.g., tema_period, cmf_window).
* Task 1.3: Integrate optuna. Create a basic objective function that runs backtest_pair for a single symbol (e.g., BTCUSDT) with parameters suggested by optuna. The objective should return a single performance metric (e.g., Sharpe Ratio or your custom score from that single backtest).
* Task 1.4: Run a small optuna study (e.g., 20-30 trials) for that single symbol. Verify that different parameters are being tested and results are stored.
Sprint 2: Multi-Symbol Optimization & Walk-Forward Structure
* Task 2.1: Modify the optuna objective function (or the function it calls) to run backtests for a small set of representative symbols (e.g., 3-5 diverse pairs) for each trial. The objective function should then return an aggregate metric (e.g., average Sharpe, median score).
* Task 2.2: Design the data splitting logic for walk-forward analysis (functions to get IS and OOS date ranges/data slices).
* Task 2.3: Implement the main walk-forward loop structure. For each IS period, it will call the optuna study (from Task 2.1, possibly with fewer trials per WFA step to save time).
* Task 2.4: For each OOS period, run a single backtest using the best parameters found from the preceding IS period. Store OOS performance.
Sprint 3: Full Walk-Forward Analysis & Result Aggregation
* Task 3.1: Run the complete walk-forward analysis over your entire dataset and multiple IS/OOS segments. This will be computationally intensive.
* Task 3.2: Develop scripts/notebooks to aggregate and analyze the OOS performance metrics. Visualize equity curves from OOS periods.
* Task 3.3: Evaluate the stability and profitability of the strategy based on aggregated OOS results.
Sprint 4 (Future): Towards Live Implementation (if desired)
* Task 4.1: Enhance BitgetAPI for authenticated trading endpoints (place order, cancel order, get balance, etc.).
* Task 4.2: Develop order management logic.
* Task 4.3: Implement robust real-time signal monitoring and execution.
* Task 4.4: Set up paper trading.
Tools & Libraries:
optuna: For parameter optimization.
pandas: For data manipulation and slicing.
numpy: For numerical operations.
matplotlib/seaborn/plotly: For visualizing results (equity curves, parameter importance).
multiprocessing / concurrent.futures: You're already using this for scan_pairs. It will be essential for speeding up optimization trials and walk-forward steps.