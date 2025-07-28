**F1 Pit Stop Strategy Predictor**

This project predicts when a Formula 1 car will pit based on historical race data.  
It’s built using Python and FastF1, with a LightGBM model trained to forecast the lap the racer's next pit stop will occur on.  
The goal is to uncover patterns in race strategy and improve predictive modeling in motorsport.

---

## Project Goals

- Predict the lap the racer's next pit stop will occur on  
- Explore how tire compounds, stint behavior, and track-specific patterns influence pit strategy  
- Build a model that generalizes across different races and teams

---

## Tech Stack

- **Data:** FastF1, pandas  
- **Modeling:** LightGBM, scikit-learn  
- **Visualization:** matplotlib  

---

## 📊 Methodology

### Data Collection

FastF1 pulls full race data for each event, including stint data, lap times, compound types, and more.

### Feature Engineering

- Current tire compound  
- Stint number  
- Is current lap time a PB? 
- Lap Number
- Track Status (flags/safety car, etc.)

### Target Variable

- The lap number at which the next pit stop will occur

### Train/Test Strategy

- Train/test split is done by **race**, not by lap, to avoid information leakage  
- Evaluated with MAE and % of predictions within ±1 lap

### Overfitting Mitigation

- Initially overfit due to non-independent test laps  
- Now split by race and pruned features to reduce race-specific bias

---

## 📈 Sample Output

![Sample Model Output](https://github.com/k-dickinson/f1-pit-strat-model/blob/main/F1_sample_output.png)

- Visual feature importance  
- Line plot comparing predicted vs actual pit lap

---

## Future Improvements

- Add circuit-level metadata (e.g., average pit delta, tire degradation curves)  
- Train on multi-year data for more robustness  
- Try sequence models (e.g. LSTM) for longer-term strategy prediction  
- Build a real-time race simulation overlay

---

## Why This Matters

Modeling pit strategies is one of the hardest problems in motorsport analytics.  
It combines tactical planning, tire wear, race conditions, and even driver style.  
This project shows how machine learning can help decode part of that complexity.

---

## 📬 Contact

Built by [@k-dickinson](https://github.com/k-dickinson) – always open to feedback or ideas!
Instagram: [@Quant_Kyle](https://instagram.com/quant_kyle)
