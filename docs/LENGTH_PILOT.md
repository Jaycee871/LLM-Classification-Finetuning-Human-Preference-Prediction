# Second experiment: length-only prediction with inner-fold tuning

After the first official-data CPU TF-IDF model underperformed a constant-prior baseline, this exploratory experiment asks whether response lengths alone offer simple, reproducible predictive signal.

Use the same seeded 12,000-row stratified sample and same 15% outer validation set for **comparability with the first pilot**. On the remaining 85%, hold out another 20% for inner hyperparameter selection from C = {0.01, 0.1, 1.0, 10.0}, and refit the selected length-only logistic-regression model on the full outer training partition.

Predictors comprise signed log character-length difference between responses, absolute length difference, average response log length, log prompt length, and two length interaction/ratio features. Augment the **training-only** rows with reversed A/B pairs and reversed winner labels. Report outer-fold multiclass log loss against uniform and training-class-prior baselines, and row-bootstrap 95% CI for the per-example excess log loss. This is observational; it does not demonstrate causal preference for verbosity, stylistic warmth, or model quality.

**Important**: Repeated outer-fold comparisons during model development make these exploratory results unsuitable as independent final evidence. Later create new prompt-grouped held-out tests and run replication across seeds.

The GitHub workflow `official-length-pilot.yml` downloads official data to an ephemeral runner; uploads **aggregate metrics JSON only**. Its first merge-triggered run is complete. The workflow is now **manual-dispatch only** so maintenance commits do not unexpectedly download data. There is no Kaggle leaderboard submission and no GPU training in this workflow.
