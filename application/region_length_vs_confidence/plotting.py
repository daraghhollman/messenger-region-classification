"""
A script to plot data from get_points.py, to try find a physically reasonable minimum crossing duration.

Region confidence is described by:

    1 - Mean( Probability Ratio within the region )

where:

    Probability Ratio = second largest probability / largest probability

for each classification within the region. i.e. each second.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import kneed
import scipy.optimize

wong_colours = {
    "black": "black",
    "orange": "#E69F00",
    "light blue": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "pink": "#CC79A7",
}

# Load data
region_metrics = pd.read_csv(
    "/home/daraghhollman/Main/Work/mercury/Code/MESSENGER_Region_Detection/application/region_length_vs_confidence/region_metrics.csv"
)

# Find knee point
kneedle = kneed.KneeLocator(region_metrics["Duration (seconds)"], region_metrics["Confidence (Mean Method)"], curve="concave", direction="increasing")

def Log_Fit(x, a, b, c):
    return 1 - np.exp(- a * (x - b)) + c

test_pars = [1, 1, 1]
pars, cov = scipy.optimize.curve_fit(Log_Fit, region_metrics["Duration (seconds)"], region_metrics["Confidence (Mean Method)"], test_pars)
fit_errors = np.sqrt(np.diag(cov))

fig, ax = plt.subplots()

ax.scatter(
    region_metrics["Duration (seconds)"],
    region_metrics["Confidence (Mean Method)"],
    marker=".",
    color="black",
    label=f"N={len(region_metrics)}",
)

x_range = np.linspace(1, 1000, 1000)
ax.plot(x_range, Log_Fit(x_range, *pars), color=wong_colours["blue"], label=r"Least Squares Fit: f(x) = $1 - e^{-a(x - b)} + c$" + f",\n    $a = {pars[0]:.3f} \pm {fit_errors[0]:.3f}$,\n    $b = {pars[1]:.0f} \pm {fit_errors[1]:.0f}$,\n    $c = {pars[2]:.2f} \pm {fit_errors[2]:.2f}$")

ax.axvline(kneedle.knee, color=wong_colours["red"], label=f"Curve Knee = {kneedle.knee} seconds")
ax.axhline(Log_Fit(kneedle.knee, *pars), color=wong_colours["orange"], label=f"Fit @ Curve Knee = {Log_Fit(kneedle.knee, *pars):.2f}")

ax.set_xlabel("Region Duration [seconds]")
ax.set_ylabel("Region Confidence (Mean Method) [arb.]")

ax.margins(0)

ax.legend()

# ax.set_xlim(0, 100)

plt.show()
