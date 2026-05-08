# Bonus: Beta Sweep Mini-Experiment

I compared three DPO beta values: 0.05, 0.1, and 0.5. Beta controls how strongly the DPO-trained policy is constrained against the reference model. A lower beta allows stronger preference optimization, while a higher beta keeps the model closer to the original SFT model.

In this mini-experiment, beta = 0.1 gave the strongest balance between reward gap and sampled win rate. Beta = 0.05 showed more aggressive movement, which can improve preference separation but may increase instability or overfitting. Beta = 0.5 was more conservative and stayed closer to the reference model, but the reward gap and win rate were weaker.

This result is consistent with the expected DPO trade-off: too little constraint can make the policy move too strongly, while too much constraint can underfit the preference signal. For a small Colab T4 run, beta = 0.1 appears to be the most practical setting.
