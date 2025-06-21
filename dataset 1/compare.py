import pandas as pd
import numpy as np

sub1 = pd.read_csv("submission.csv")  # replace with actual filename
sub2 = pd.read_csv("submission1.csv")

# Check how correlated they are
corr = np.corrcoef(sub1["smoking"], sub2["smoking"])[0, 1]
print("Correlation between submissions:", corr)

# Check disagreement
disagree = np.mean((sub1["smoking"] > 0.5) != (sub2["smoking"] > 0.5))
print("Percentage of differing predictions:", disagree * 100, "%")
