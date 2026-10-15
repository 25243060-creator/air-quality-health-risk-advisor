"""
Air Quality Health Risk Advisor - Bayesian network (CO10)
Run:  streamlit run aqi_app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
from aqi_engine import (VARS, exact, likelihood_weighting, naive_bayes_fit, naive_bayes_predict,
                        sample_data, policy_effects, advice, compare_models)

st.set_page_config(page_title="Air Quality Health Risk Advisor", page_icon="😷", layout="wide")

LABELS = {"Season": "Season", "DayType": "Day type", "Rain": "Raining today?", "Wind": "Wind",
          "Traffic": "Traffic nearby", "Industry": "Factory / industry nearby?",
          "Burning": "Waste / crop burning nearby?", "Haze": "Sky / visibility",
          "Breathless": "Feeling breathless / cough?", "Eyes": "Eye irritation?",
          "Group": "Child / elderly / breathing problem?"}
PRESETS = {
    "(none)": {},
    "Winter morning near highway": {"Season": "Winter", "Wind": "Calm", "Traffic": "High", "Haze": "Hazy"},
    "Post-harvest burning season": {"Season": "Winter", "Burning": "Yes", "Wind": "Calm", "Haze": "Hazy",
                                    "Eyes": "Yes", "Breathless": "Yes"},
    "Rainy monsoon day": {"Season": "Monsoon", "Rain": "Yes", "Wind": "Windy", "Haze": "Clear"},
    "Summer afternoon, windy": {"Season": "Summer", "Wind": "Windy", "Traffic": "Low", "Haze": "Clear"},
}


def apply_preset():
    p = PRESETS[st.session_state["preset"]]
    for v in LABELS:
        st.session_state[f"in-{v}"] = p.get(v, "Unknown")


st.title("😷 Air Quality Health Risk Advisor")
st.caption("A Bayesian network that reasons under uncertainty: from what you can see and feel, it estimates "
           "the hidden air-quality level (AQI) and your health risk, and suggests what to do.")

with st.sidebar:
    st.header("Settings")
    method = st.radio("Inference method", ["Exact (enumeration)", "Approximate (likelihood weighting)",
                                            "Naive Bayes (AQI only)"])
    n_samples = st.slider("Samples", 500, 20000, 5000, 500, disabled=not method.startswith("Approx"))
    st.selectbox("Load an example", list(PRESETS), key="preset")
    st.button("Apply example", on_click=apply_preset)
    st.markdown("---")
    st.markdown("Leave anything you don't know as **Unknown**: the network still reasons with what you give.")

# ---------------- 1. INPUTS ----------------
st.subheader("1. What do you know?")
cols = st.columns(4)
evidence = {}
for i, v in enumerate(LABELS):
    with cols[i % 4]:
        val = st.selectbox(LABELS[v], ["Unknown"] + VARS[v], key=f"in-{v}")
        if val != "Unknown":
            evidence[v] = val


@st.cache_resource
def nb_model():
    return naive_bayes_fit(sample_data(20000, seed=1))


def infer(query, ev, do=None):
    if method.startswith("Exact"):
        return exact(query, ev, do)
    if method.startswith("Approx"):
        return likelihood_weighting(query, ev, do, n=n_samples)
    if query == "AQI" and not do:
        return naive_bayes_predict(nb_model(), ev)
    return exact(query, ev, do)       # naive Bayes has no Risk node: fall back to exact


# ---------------- 2. RESULTS ----------------
st.subheader("2. Results")
aqi_p, risk_p = infer("AQI", evidence), infer("Risk", evidence)
c1, c2, c3 = st.columns([2, 2, 3])
with c1:
    st.markdown("**Air quality (AQI) probability**")
    st.bar_chart(pd.DataFrame({"P": aqi_p}, index=VARS["AQI"]))
with c2:
    st.markdown("**Health risk probability**")
    st.bar_chart(pd.DataFrame({"P": risk_p}, index=VARS["Risk"]))
with c3:
    level, tip = advice(risk_p)
    st.metric("Most likely AQI", VARS["AQI"][int(np.argmax(aqi_p))], f"{aqi_p.max()*100:.0f}%")
    (st.error if level == "HIGH RISK" else st.warning if level == "MEDIUM RISK" else st.success)(f"**{level}**: {tip}")
    if method.startswith("Naive"):
        st.caption("Naive Bayes only predicts AQI; the risk is computed from the Bayesian network.")

# ---------------- 3. CAUSAL WHAT-IF ----------------
st.subheader("3. Causal what-if: what if the government acts? (do-operator)")
rows = policy_effects(evidence)
df = pd.DataFrame(rows).set_index("Policy")
st.dataframe((df * 100).round(1).astype(str) + " %")
st.caption("do(Traffic = Low) etc. cuts the arrows into that variable and fixes its value, "
           "so we see the effect of the action itself, not just correlation.")

st.markdown("**Seeing vs doing**")
a, b, c = st.columns(3)
a.metric("Prior P(Unhealthy)", f"{exact('AQI')[2]*100:.1f}%")
b.metric("Observe: I feel breathless", f"{exact('AQI', {'Breathless': 'Yes'})[2]*100:.1f}%")
c.metric("Do: breathless from a cold (cause cut)", f"{exact('AQI', {}, {'Breathless': 'Yes'})[2]*100:.1f}%")
st.caption("Symptoms caused by the air tell us about the air. Symptoms caused by something else (do-operator) tell us nothing.")

# ---------------- 4. EXACT vs APPROXIMATE ----------------
st.subheader("4. Exact vs approximate inference")
ex, ap = exact("AQI", evidence), likelihood_weighting("AQI", evidence, n=n_samples)
t1, t2, t3 = st.columns(3)
t1.metric("P(Unhealthy) exact", f"{ex[2]*100:.2f}%")
t2.metric(f"P(Unhealthy) LW ({n_samples})", f"{ap[2]*100:.2f}%")
t3.metric("Max error", f"{np.abs(ex - ap).max()*100:.2f}%")

# ---------------- 5. MODEL COMPARISON ----------------
st.subheader("5. Naive Bayes vs Bayesian network accuracy")
if st.button("Run evaluation (3000 test cases)"):
    with st.spinner("Evaluating..."):
        res = pd.DataFrame(compare_models()).set_index("Evidence").round(1)
    st.dataframe(res)
    st.bar_chart(res)

# ---------------- 6. NETWORK ----------------
st.subheader("6. Bayesian network structure")
st.graphviz_chart("""digraph { rankdir=LR; node [shape=ellipse];
DayType->Traffic; Season->Rain; Season->Wind; Season->Burning;
Traffic->AQI; Industry->AQI; Burning->AQI; Wind->AQI; Rain->AQI;
AQI->Haze; AQI->Breathless; AQI->Eyes; AQI->Risk; Group->Risk; }""")
