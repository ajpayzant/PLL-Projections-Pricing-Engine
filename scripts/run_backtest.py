"""
Run walk-forward backtest on 2023-2025 seasons and print results.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from projection_engine_v2 import DataLoader, Backtester
import pandas as pd

loader = DataLoader()
bt = Backtester(loader, n_sims=5000)

print("Running walk-forward backtest on 2023-2025 seasons...")
print("(This fits a fresh model before each game — takes a few minutes)")
print()

result = bt.run(val_seasons=[2023, 2024, 2025])

print(f"Games evaluated:    {result.n_games}")
print(f"MAE home goals:     {result.mae_home_goals:.3f}")
print(f"MAE away goals:     {result.mae_away_goals:.3f}")
print(f"MAE total goals:    {result.mae_total:.3f}")
print(f"RMSE total goals:   {result.rmse_total:.3f}")
print(f"Brier score:        {result.brier_score:.4f}")
print()
print("Calibration table (predicted win prob bucket vs actual win rate):")
print(result.calibration_table.to_string(index=False))
