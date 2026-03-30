Second-round YOLOv8 upgrade package

Why this round exists
The first round showed:
- Tooth is very easy
- Caries / Cavity / Crack are much weaker
- Many lesion instances are still treated as background
- Threshold tuning helps, but does not solve the core issue

So this round focuses on:
1. lesion-heavy oversampling
2. stronger class-loss emphasis
3. stable training (multi_scale kept off)
4. preserving the same val/test split for fair comparison

Files
- build_rebalanced_train_list.py
  Create a train txt list that repeats lesion-containing images more often.
- train_second_round.py
  Train a second-round model on the rebalanced train list.
- run_second_round_example.txt
  Copy-paste commands.

Recommended repeat map
- class 0 (Caries): 2
- class 1 (Cavity): 2
- class 2 (Crack): 4
- class 3 (Tooth): no special repeat, keep background_repeat=1

Notes
- Keep the same original val/test.
- Compare second round to first round using:
  - same val split
  - same threshold sweep process
  - same metrics
- Do NOT change too many things at once beyond this package.
