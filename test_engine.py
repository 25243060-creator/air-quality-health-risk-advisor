import time
import numpy as np
from aqi_engine import *

t0 = time.time()
n = lambda a: " ".join("%s=%.3f" % (s, p) for s, p in zip(VARS["AQI"], a))
print("Prior AQI           :", n(exact("AQI")))
ev = {"Season": "Winter", "Wind": "Calm", "Traffic": "High", "Haze": "Hazy", "Breathless": "Yes", "Group": "Yes"}
print("Winter/calm/traffic/haze/breathless/sensitive group:")
e = exact("AQI", ev); l = likelihood_weighting("AQI", ev, n=20000, seed=1)
print("  AQI exact         :", n(e))
print("  AQI LW (20000)    :", n(l), " max error %.4f" % np.abs(e - l).max())
r = exact("Risk", ev)
print("  Risk exact        : Low=%.3f Medium=%.3f High=%.3f -> %s" % (*r, advice(r)[0]))
print("Policy what-if (do-operator):")
for row in policy_effects(ev):
    print("  %-34s P(Unhealthy) %.3f -> %.3f | P(High risk) %.3f -> %.3f" % (
        row["Policy"], row["P(Unhealthy) before"], row["P(Unhealthy) after"],
        row["P(High risk) before"], row["P(High risk) after"]))
print("Seeing vs doing (symptom):")
print("  observe Breathless=Yes : P(Unhealthy) = %.3f" % exact("AQI", {"Breathless": "Yes"})[2])
print("  do(Breathless=Yes)     : P(Unhealthy) = %.3f" % exact("AQI", {}, {"Breathless": "Yes"})[2])
print("  prior                  : P(Unhealthy) = %.3f" % exact("AQI")[2])
print("Model comparison (predict AQI category, 3000 test cases):")
for row in compare_models():
    print("  %-56s NB %.1f%% | BN exact %.1f%% | BN LW %.1f%%" % (
        row["Evidence"], row["Naive Bayes %"], row["BN exact %"], row["BN LW % (300 cases)"]))
print("time %.1fs" % (time.time() - t0))
