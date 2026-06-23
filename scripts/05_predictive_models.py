"""
LAYER 4 — PREDICTIVE MODELLING
===============================
Predicts game outcome (win/loss) from team performance differentials.
Satisfies the assignment's "at least one modelling technique" requirement
with three models compared rigorously.

Models:
  - Logistic Regression (interpretable, linear)
  - Decision Tree (interpretable, non-linear, gives rules)
  - Random Forest (high accuracy, feature importance)

Validation: 5-fold stratified cross-validation + held-out test set
Metrics: accuracy, ROC-AUC, precision, recall, F1
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, classification_report,
                             confusion_matrix, roc_curve)

df = pd.read_csv('/home/claude/team_game_features.csv')

# Features: the four differentials + home advantage
features = ['assist_diff','reb_diff','steal_diff','block_diff','isHome']
X = df[features]
y = df['win']

print(f"Dataset: {len(df):,} observations, {len(features)} features")
print(f"Win rate: {y.mean():.3f}\n")

# Train / test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree (depth=4)': DecisionTreeClassifier(max_depth=4, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=8,
                                            random_state=42, n_jobs=-1),
}

print("="*70)
print("MODEL COMPARISON (5-fold CV on training set + test set)")
print("="*70)
print(f"{'Model':<26}{'CV Acc':>10}{'Test Acc':>10}{'Test AUC':>10}")
print("-"*70)

results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {'model':model,'acc':acc,'auc':auc,'cv':cv_scores.mean()}
    print(f"{name:<26}{cv_scores.mean():>9.3f}{acc:>10.3f}{auc:>10.3f}")

# ---------------------------------------------------------------
# Best model details
# ---------------------------------------------------------------
print("\n" + "="*70)
print("LOGISTIC REGRESSION — coefficients (log-odds)")
print("="*70)
lr = results['Logistic Regression']['model']
for f, c in zip(features, lr.coef_[0]):
    print(f"  {f:<14} coef={c:+.4f}   odds_ratio={np.exp(c):.3f}")

# ---------------------------------------------------------------
# Feature importance (Random Forest)
# ---------------------------------------------------------------
print("\n" + "="*70)
print("FEATURE IMPORTANCE (Random Forest)")
print("="*70)
rf = results['Random Forest']['model']
imp = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
for f, v in imp.items():
    print(f"  {f:<14} {v:.4f}  {'#'*int(v*50)}")

# ---------------------------------------------------------------
# Decision Tree rules (human-readable for the coach)
# ---------------------------------------------------------------
print("\n" + "="*70)
print("DECISION TREE RULES (top levels — coach-readable)")
print("="*70)
dt = results['Decision Tree (depth=4)']['model']
tree_rules = export_text(dt, feature_names=features, max_depth=3)
print(tree_rules)

# ---------------------------------------------------------------
# Confusion matrix for best model
# ---------------------------------------------------------------
best_name = max(results, key=lambda k: results[k]['auc'])
print("="*70)
print(f"BEST MODEL: {best_name}")
print("="*70)
best = results[best_name]['model']
y_pred = best.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion matrix:")
print(f"               Pred Loss  Pred Win")
print(f"  Actual Loss   {cm[0,0]:>7}   {cm[0,1]:>7}")
print(f"  Actual Win    {cm[1,0]:>7}   {cm[1,1]:>7}")
print(f"\n{classification_report(y_test, y_pred, target_names=['Loss','Win'])}")

# Save ROC data for visualization
roc_data = {}
for name in results:
    m = results[name]['model']
    proba = m.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_data[name] = (fpr.tolist(), tpr.tolist(), results[name]['auc'])

import json
with open('/home/claude/roc_data.json','w') as f:
    json.dump({'roc':roc_data,
               'importance':imp.to_dict(),
               'lr_coef':dict(zip(features, lr.coef_[0].tolist()))}, f)
print("\n[Saved model outputs -> roc_data.json]")
