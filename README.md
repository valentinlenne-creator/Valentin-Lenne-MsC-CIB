# Master's Thesis — MSc Corporate and Investment Banking — NEOMA Business School

## Combining Fundamental Valuation Criteria and Risk-Adjusted Performance 
## Indicators for the Assessment of Equity Portfolios

This repository contains the complete Python code developed as part of my 
master's thesis.

### Description

The script builds and compares three equity portfolios from a universe of 
348 global companies over a three-year period:
- Portfolio 1: pure fundamental approach
- Portfolio 2: pure risk-adjusted performance approach
- Portfolio 3: combined approach

It also runs three OLS regressions testing the explanatory power of the 
different families of indicators on the three-horizon Sharpe ratio.

### Required libraries
numpy
pandas
yfinance
scipy
scikit-learn
statsmodels
plotly
Installation : `pip install numpy pandas yfinance scipy scikit-learn statsmodels plotly`

### Exécution
python PORTEFEUILLE_MEMOIRE.py
### Data sources

Market data retrieved via Yahoo Finance (yfinance library).

### Note

The `par84` module imported in the script handles only the visual formatting 
of the charts and does not affect any of the calculations or results.

### Author

Valentin Lenne — MSc Corporate and Investment Banking — NEOMA Business School — 2025-2026
