# Does More Passing Win Basketball Games? 🏀

**An analysis of winning conditions in professional basketball using 11 seasons of NBA data (2009–2020).**

Sports Data Science — Final Project · Leiden University (LIACS)

---

## Overview

This project answers a real coaching question posed by Bert Samson, head coach of the basketball club *The Hague Royals*:

> *"Does more passing in a set offence increase the chance of winning?"*

and the broader question: *"What do we need to do to win?"*

Using a dataset of 12,777 NBA games, the analysis combines **hypothesis testing** (confirming the coach's theories) with **hypothesis generation** (discovering new winning patterns from the data).

The findings were shared with Coach Samson, who reviewed the report and provided feedback that is incorporated into the analysis.

## Key findings

| Finding | Result |
|---|---|
| Assist differential is the single strongest predictor of winning | 44% feature importance |
| Dominating 3 of 4 statistical categories | 81% win rate |
| Dominating all 4 categories | 95% win rate |
| Three-pointers that are assisted (created by a pass) | 83% |
| Teams with 9+ players recording assists | 56% win rate |
| Predictive model accuracy (logistic regression, 5 features) | 79%, AUC 0.87 |

The assist→win effect is partly mechanical (assists require baskets), but a genuine **playmaking signal persists** even after controlling for scoring — confirming the coach's hypothesis with appropriate nuance.

## Repository structure
