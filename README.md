# Does More Passing Win Basketball Games? 🏀

**An analysis of winning conditions in professional basketball using 11 seasons of NBA data (2009–2020).**

Sports Data Science — Final Project · Leiden University (LIACS)

---

## Overview

This project answers a real coaching question posed by Bert Samson, head coach of the basketball club *The Hague Royals*:

> *"Does more passing in a set offence increase the chance of winning?"*

and the broader question: *"What do we need to do to win?"*

Using a dataset of 12,777 NBA games, the analysis combines **hypothesis testing** (confirming the coach's theories) with **hypothesis generation** (discovering new winning patterns from the data).

## Key findings

| Finding | Result |
|---|---|
| Assist differential is the single strongest predictor of winning | 44% feature importance |
| Dominating 3 of 4 statistical categories | 83% win rate |
| Dominating all 4 categories | 95% win rate |
| Three-pointers that are assisted (created by a pass) | 81% |
| Teams with 9+ players recording assists | 56% win rate |
| Predictive model accuracy (logistic regression, 5 features) | 79%, AUC 0.87 |

The assist→win effect is partly mechanical (assists require baskets), but a genuine **playmaking signal persists** even after controlling for scoring — confirming the coach's hypothesis with appropriate nuance.

## Repository structure

```
├── scripts/
│   ├── 00_extract_playbyplay.py        # Extract assist context from 300MB play-by-play
│   ├── 01_descriptive_hypothesis.py    # Q0 & Q3 exploration
│   ├── 02_statistical_tests.py         # Significance testing (t-test, chi-square, logistic)
│   ├── 03_circularity_control.py       # Endogeneity / circularity analysis
│   ├── 04_subgroup_discovery.py        # Hypothesis generation (pysubgroup)
│   ├── 05_predictive_models.py         # Logistic regression, decision tree, random forest
│   ├── 06_quarter_momentum.py          # Quarter-by-quarter momentum analysis
│   ├── 07_season_trends.py             # Robustness & three-point revolution
│   ├── 08_player_role_distribution.py  # Player role, assist distribution, injuries
│   └── 09_generate_figures.py          # Publication-quality figure generation
├── figures/                            # All generated figures (PNG)
├── docs/                               # Report and dashboard
└── README.md
```

## Methods

The analysis is structured in five layers:

1. **Descriptive exploration** — distributions, win-rate patterns
2. **Statistical hypothesis testing** — point-biserial correlation, Welch's t-test, chi-square, logistic regression with effect sizes (Cohen's d, Cramér's V, odds ratios)
3. **Subgroup discovery** — automated discovery of high/low win-rate conditions (WRAcc beam search)
4. **Predictive modelling** — three classifiers compared with 5-fold cross-validation
5. **Coach dashboard** — actionable, interpretable summary

## Data

NBA data (2009–2020), three linked tables:
- `match_data_dump.csv` — 12,777 games, team-level statistics
- `player_data_dump.csv` — 351,110 player-game records
- `playbyplay_data_dump.csv` — 2.1M+ play-by-play events (~300MB)

*Data provided by the course; not redistributed here.*

## Tech stack

`pandas` · `numpy` · `scipy` · `statsmodels` · `scikit-learn` · `pysubgroup` · `matplotlib`

## Author
Emre Ulusoy — Sports Data Science, Leiden University (LIACS), 2026
MSc Student of Data Science- Leiden University
Former university-level basketball and football player (Turkey).
