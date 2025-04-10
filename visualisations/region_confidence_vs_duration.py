
import pandas as pd
import matplotlib.pyplot as plt

hollman_regions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_regions.csv"
)

print(hollman_regions.loc[hollman_regions["Duration (seconds)"] < 0])

1/0

fig, ax = plt.subplots()

ax.scatter(hollman_regions["Duration (seconds)"], hollman_regions["Confidence"], marker=".")

plt.show()
