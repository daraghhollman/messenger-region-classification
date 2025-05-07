
import numpy as np
import scipy.optimize
import pandas as pd
import matplotlib.pyplot as plt
import kneed
from hermpy.plotting import wong_colours

regions = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/data/new_regions.csv"
)

# Find knee point
kneedle = kneed.KneeLocator(regions["Duration (seconds)"], regions["Confidence"], curve="concave", direction="increasing")

def Log_Fit(x, a, b, c):
    return 1 - np.exp(- a * (x - b)) + c

test_pars = [1, 1, 1]
pars, cov = scipy.optimize.curve_fit(Log_Fit, regions["Duration (seconds)"], regions["Confidence"], test_pars)
fit_errors = np.sqrt(np.diag(cov))

fig, ax = plt.subplots()

x_range = np.linspace(1, 10000, 1000)
ax.plot(x_range, Log_Fit(x_range, *pars), color=wong_colours["blue"], label=r"Least Squares Fit: f(x) = $1 - e^{-a(x - b)} + c$" + f",\n    $a = {pars[0]:.3f} \pm {fit_errors[0]:.3f}$,\n    $b = {pars[1]:.0f} \pm {fit_errors[1]:.0f}$,\n    $c = {pars[2]:.2f} \pm {fit_errors[2]:.2f}$")

ax.axvline(kneedle.knee, color=wong_colours["red"], label=f"Curve Knee = {kneedle.knee} seconds")
ax.axhline(Log_Fit(kneedle.knee, *pars), color=wong_colours["orange"], label=f"Fit @ Curve Knee = {Log_Fit(kneedle.knee, *pars):.2f}")

ax.set_xlabel("Region Duration [seconds]")
ax.set_ylabel("Region Confidence [arb.]")

ax.margins(0)

ax.legend()

ax.hist2d(regions["Duration (seconds)"], regions["Confidence"], norm="log", bins=100)

ax.margins(0)

plt.show()
