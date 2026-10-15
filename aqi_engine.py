"""
aqi_engine.py - Bayesian network engine for the Air Quality Health Risk Advisor (CO10).

Network (13 discrete variables):
  DayType -> Traffic
  Season  -> Rain, Wind, Burning
  Traffic, Industry, Burning, Wind, Rain -> AQI
  AQI -> Haze, Breathless, Eyes
  AQI, Group -> Risk

Covers: Bayes inference, naive Bayes, exact inference (full-joint enumeration),
approximate inference (likelihood weighting), causal do() intervention.
"""
import itertools
import numpy as np

# ---------------------------------------------------------------- structure
VARS = {
    "DayType": ["Weekday", "Weekend"],
    "Season": ["Winter", "Summer", "Monsoon"],
    "Rain": ["No", "Yes"],
    "Wind": ["Calm", "Windy"],
    "Burning": ["No", "Yes"],          # crop / garbage burning nearby
    "Traffic": ["Low", "High"],
    "Industry": ["No", "Yes"],         # factory nearby
    "AQI": ["Good", "Moderate", "Unhealthy"],
    "Haze": ["Clear", "Hazy"],
    "Breathless": ["No", "Yes"],
    "Eyes": ["No", "Yes"],             # eye irritation
    "Group": ["No", "Yes"],            # child / elderly / breathing problem
    "Risk": ["Low", "Medium", "High"],
}
PARENTS = {
    "DayType": [], "Season": [], "Rain": ["Season"], "Wind": ["Season"],
    "Burning": ["Season"], "Traffic": ["DayType"], "Industry": [],
    "AQI": ["Traffic", "Industry", "Burning", "Wind", "Rain"],
    "Haze": ["AQI"], "Breathless": ["AQI"], "Eyes": ["AQI"],
    "Group": [], "Risk": ["AQI", "Group"],
}
ORDER = list(VARS)                      # topological order
OBSERVABLE = ["Season", "DayType", "Rain", "Wind", "Traffic", "Industry",
              "Burning", "Haze", "Breathless", "Eyes", "Group"]


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def _binary(p_true):
    p = np.array(p_true, dtype=float)
    return np.stack([1 - p, p], axis=-1)


def build_cpts():
    cpt = {}
    cpt["DayType"] = np.array([0.7, 0.3])
    cpt["Season"] = np.array([0.3, 0.4, 0.3])
    cpt["Rain"] = _binary([0.10, 0.15, 0.70])        # P(Rain=Yes | Season)
    cpt["Wind"] = _binary([0.30, 0.50, 0.60])        # P(Windy | Season)
    cpt["Burning"] = _binary([0.40, 0.10, 0.05])     # P(Burning=Yes | Season)
    cpt["Traffic"] = _binary([0.70, 0.35])           # P(High | Weekday/Weekend)
    cpt["Industry"] = np.array([0.7, 0.3])
    # AQI: ordered-logit style CPT from a pollution score
    aqi = np.zeros((2, 2, 2, 2, 2, 3))
    for t, i, b, w, r in itertools.product(range(2), repeat=5):
        s = 0.9 * t + 1.1 * i + 1.4 * b - 0.9 * w - 1.3 * r
        p_mod_up = _sig(2 * (s - 0.2))               # P(AQI >= Moderate)
        p_unh = _sig(2 * (s - 1.6))                  # P(AQI = Unhealthy)
        aqi[t, i, b, w, r] = [1 - p_mod_up, p_mod_up - p_unh, p_unh]
    cpt["AQI"] = aqi
    cpt["Haze"] = _binary([0.05, 0.35, 0.85])        # P(Hazy | AQI)
    cpt["Breathless"] = _binary([0.05, 0.20, 0.55])
    cpt["Eyes"] = _binary([0.08, 0.25, 0.60])
    cpt["Group"] = np.array([0.75, 0.25])
    cpt["Risk"] = np.array([                          # [AQI][Group] -> (Low, Med, High)
        [[0.90, 0.09, 0.01], [0.70, 0.25, 0.05]],
        [[0.60, 0.30, 0.10], [0.20, 0.50, 0.30]],
        [[0.20, 0.45, 0.35], [0.02, 0.28, 0.70]],
    ])
    return cpt


CPT = build_cpts()
_LET = {v: chr(97 + i) for i, v in enumerate(ORDER)}


def _state(v, s):
    return s if isinstance(s, (int, np.integer)) else VARS[v].index(s)


# ---------------------------------------------------------------- joint distribution
def joint(do=None):
    """Full joint tensor P(all variables). do={var: state} cuts the incoming edges
    of var and fixes its value (truncated factorisation)."""
    do = do or {}
    subs, ops = [], []
    for v in ORDER:
        if v in do:
            f = np.zeros(len(VARS[v]))
            f[_state(v, do[v])] = 1.0
            subs.append(_LET[v])
            ops.append(f)
        else:
            subs.append("".join(_LET[p] for p in PARENTS[v]) + _LET[v])
            ops.append(CPT[v])
    return np.einsum(",".join(subs) + "->" + "".join(_LET[v] for v in ORDER), *ops, optimize=True)


_J0 = joint()


# ---------------------------------------------------------------- EXACT INFERENCE
def exact(query, evidence=None, do=None):
    """P(query | evidence, do(do)) by enumerating the full joint. Returns array over states."""
    evidence = {k: v for k, v in (evidence or {}).items() if not do or k not in do}
    J = joint(do) if do else _J0
    idx = [slice(None)] * len(ORDER)
    for v, s in evidence.items():
        idx[ORDER.index(v)] = _state(v, s)
    J2 = J[tuple(idx)]
    remaining = [v for v in ORDER if v not in evidence]
    if query in evidence:
        out = np.zeros(len(VARS[query]))
        out[_state(query, evidence[query])] = 1.0
        return out
    q = remaining.index(query)
    marg = J2.sum(axis=tuple(i for i in range(len(remaining)) if i != q))
    return marg / marg.sum()


# ---------------------------------------------------------------- APPROXIMATE INFERENCE
def likelihood_weighting(query, evidence=None, do=None, n=5000, seed=None):
    rng = np.random.default_rng(seed)
    evidence, do = evidence or {}, do or {}
    vals, w = {}, np.ones(n)
    for v in ORDER:
        k = len(VARS[v])
        if v in do:
            vals[v] = np.full(n, _state(v, do[v]))
            continue
        probs = CPT[v][tuple(vals[p] for p in PARENTS[v])] if PARENTS[v] else np.tile(CPT[v], (n, 1))
        if v in evidence:
            e = _state(v, evidence[v])
            vals[v] = np.full(n, e)
            w *= probs[:, e]
        else:
            c = probs.cumsum(axis=1)
            vals[v] = np.minimum((rng.random(n)[:, None] > c).sum(axis=1), k - 1)
    out = np.bincount(vals[query], weights=w, minlength=len(VARS[query]))
    return out / out.sum() if out.sum() > 0 else np.ones(len(VARS[query])) / len(VARS[query])


# ---------------------------------------------------------------- synthetic data + NAIVE BAYES
def sample_data(n, seed=0):
    """Forward-sample n complete rows from the network (state indices)."""
    rng = np.random.default_rng(seed)
    vals = {}
    for v in ORDER:
        k = len(VARS[v])
        probs = CPT[v][tuple(vals[p] for p in PARENTS[v])] if PARENTS[v] else np.tile(CPT[v], (n, 1))
        c = probs.cumsum(axis=1)
        vals[v] = np.minimum((rng.random(n)[:, None] > c).sum(axis=1), k - 1)
    return vals


NB_FEATURES = ["Season", "Rain", "Wind", "Traffic", "Industry", "Burning", "Haze", "Breathless", "Eyes"]


def naive_bayes_fit(data, features=NB_FEATURES, cls="AQI", alpha=1.0):
    k = len(VARS[cls])
    prior = np.bincount(data[cls], minlength=k) + alpha
    model = {"prior": prior / prior.sum(), "lik": {}}
    for f in features:
        m = len(VARS[f])
        counts = np.zeros((k, m)) + alpha
        for c in range(k):
            counts[c] = np.bincount(data[f][data[cls] == c], minlength=m) + alpha
        model["lik"][f] = counts / counts.sum(axis=1, keepdims=True)
    return model


def naive_bayes_predict(model, evidence):
    post = model["prior"].copy()
    for f, s in evidence.items():
        if f in model["lik"]:
            post = post * model["lik"][f][:, _state(f, s)]
    return post / post.sum()


# ---------------------------------------------------------------- what-if / advice
POLICIES = {
    "Reduce traffic (no-vehicle day)": ("Traffic", "Low"),
    "Ban waste / crop burning": ("Burning", "No"),
    "Shut nearby industry": ("Industry", "No"),
}


def policy_effects(evidence):
    rows = []
    for name, (var, val) in POLICIES.items():
        ev = {k: v for k, v in evidence.items() if k != var}
        b_unh = exact("AQI", evidence)[2]
        a_unh = exact("AQI", ev, {var: val})[2]
        b_high = exact("Risk", evidence)[2]
        a_high = exact("Risk", ev, {var: val})[2]
        rows.append({"Policy": name, "P(Unhealthy) before": b_unh, "P(Unhealthy) after": a_unh,
                     "P(High risk) before": b_high, "P(High risk) after": a_high})
    return rows


def advice(risk_probs):
    level = int(np.argmax(risk_probs))
    if level == 2:
        return ("HIGH RISK", "Avoid outdoor activity, wear an N95 mask if you must go out, "
                "keep windows closed and use an air purifier. Sensitive people should stay indoors.")
    if level == 1:
        return ("MEDIUM RISK", "Limit long outdoor exercise, wear a mask in traffic, "
                "and keep reliever medicines handy if you have breathing problems.")
    return ("LOW RISK", "Air is fine for normal outdoor activity. Ventilate your home.")


def compare_models(n_train=20000, n_test=3000, n_lw_test=300, seed=1):
    """Predict AQI from partial evidence: naive Bayes vs Bayesian network (exact / LW)."""
    train, test = sample_data(n_train, seed), sample_data(n_test, seed + 1)
    nb = naive_bayes_fit(train)
    scenarios = {
        "Full evidence (9 features)": NB_FEATURES,
        "Partial (Season, Wind, Traffic, Haze, Breathless, Eyes)": ["Season", "Wind", "Traffic", "Haze", "Breathless", "Eyes"],
        "Symptoms only (Haze, Breathless, Eyes)": ["Haze", "Breathless", "Eyes"],
    }
    results = []
    for name, feats in scenarios.items():
        hit = {"Naive Bayes": 0, "BN exact": 0, "BN likelihood weighting": 0}
        for i in range(n_test):
            ev = {f: int(test[f][i]) for f in feats}
            true = int(test["AQI"][i])
            hit["Naive Bayes"] += int(np.argmax(naive_bayes_predict(nb, ev)) == true)
            hit["BN exact"] += int(np.argmax(exact("AQI", ev)) == true)
            if i < n_lw_test:
                hit["BN likelihood weighting"] += int(np.argmax(likelihood_weighting("AQI", ev, n=2000, seed=i)) == true)
        results.append({
            "Evidence": name,
            "Naive Bayes %": 100 * hit["Naive Bayes"] / n_test,
            "BN exact %": 100 * hit["BN exact"] / n_test,
            "BN LW % (300 cases)": 100 * hit["BN likelihood weighting"] / n_lw_test,
        })
    return results
