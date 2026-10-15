# 😷 Air Quality Health Risk Advisor

A Bayesian-network based AI system that estimates the hidden **air quality level (AQI)** and a person's
**health risk** from uncertain, everyday clues (season, wind, rain, traffic, nearby industry, waste burning,
haze, breathlessness, eye irritation) and gives practical advice.

Mini project for **CO10 - Design and develop a mini project using AI techniques for real world applications**
(Uncertain knowledge and reasoning: Bayesian inference, naive Bayes, Bayesian networks, exact and approximate
inference, causal networks).

## Features
- **13-variable Bayesian network** with conditional probability tables (CPTs)
- **Naive Bayes** baseline
- **Exact inference** (enumeration of the full joint distribution)
- **Approximate inference** (likelihood weighting)
- **Causal what-if** with the do-operator: effect of reducing traffic, banning waste burning, shutting industry
- **Seeing vs doing**: symptoms caused by the air vs symptoms caused by something else
- Streamlit web app with example scenarios, probability charts and health advice
- Inference engine written from scratch with NumPy (no probabilistic library)

## Network structure
```
DayType -> Traffic
Season  -> Rain, Wind, Burning
Traffic, Industry, Burning, Wind, Rain -> AQI
AQI -> Haze, Breathless, Eyes
AQI, Group -> Risk
```

## Project structure
```
aqi_advisor/
├── aqi_engine.py     # network, CPTs, exact / likelihood weighting / naive Bayes / do-operator
├── aqi_app.py        # Streamlit interface
├── test_engine.py    # quick verification and evaluation
├── requirements.txt
└── README.md
```

## Setup and run
```bash
pip install -r requirements.txt
python test_engine.py            # console results
streamlit run aqi_app.py         # web app (http://localhost:8501)
```
On Windows, if `pip` or `python` points to a different Python, use `python -m pip ...` and `python -m streamlit run aqi_app.py`.

## Sample results
Case: winter, calm wind, high traffic, hazy sky, breathless, sensitive person.

| Quantity | Result |
|---|---|
| P(AQI = Unhealthy), exact | 90.3% (prior 19.4%) |
| P(AQI = Unhealthy), likelihood weighting (20000) | 90.3% (error 0.01%) |
| Health risk (Low / Medium / High) | 3.8% / 30.1% / 66.1% |
| do(Traffic = Low) | Unhealthy 90.3% -> 80.6% |
| do(Burning = No) | Unhealthy 90.3% -> 81.0% |
| do(Industry = No) | Unhealthy 90.3% -> 86.2% |

AQI prediction accuracy on 3000 test cases (all evidence): naive Bayes 76.1%, Bayesian network 78.1%.

## Limitations
CPTs are expert-style assumptions, and the accuracy tests use data sampled from the model itself, so they
compare inference methods rather than validate against real measurements. Future work: learn CPTs from real
AQI data (e.g. CPCB), add live weather feeds, and a dynamic Bayesian network for forecasting.

## Disclaimer
Educational project. Not medical advice.

## Author
Nishanthini M, National Engineering College (NEC)
