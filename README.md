## **F1 Pit Stop Strategy Predictor**

This project forecasts how many laps remain before a Formula 1 driver makes their next pit stop, based on detailed race telemetry.  
It uses Python and FastF1 to collect race data, processes it with pandas, and trains a LightGBM regression model to predict the number of laps until the next pit stop.

---

## Project Goals

- Predict **LapsToNextPit** — a dynamic label generated from historical pit timing  
- Investigate how tire compound, stint phase, track status, and lap behavior influence pitting  
- Ensure model generalizes across multiple drivers and events without race-specific leakage

---

## Tech Stack

| Category         | Tools Used            |
|------------------|------------------------|
| Data Collection  | FastF1, pandas         |
| Modeling         | LightGBM, scikit-learn |
| Evaluation       | MAE, ±1/2/3 lap accuracy |
| Visualization    | matplotlib             |

---

## Methodology

### Data Loading

- Utilizes FastF1 cache structure to load lap-level data per session  
- Supports automated ingestion across multiple races  

### Feature Engineering

- Encodes tire compound as categorical codes  
- Captures track status flags and personal best lap indicators  
- Extracts stint and lap context for predictive signals

### Label Generation

- For each driver, computes how many laps until their next recorded pit  
- Dynamically creates `LapsToNextPit` by searching forward from current lap

### Training Strategy

- Race-level split: training and testing on distinct grand prix events to prevent leakage  
- LightGBM model trained with early stopping and regularization

### Evaluation Metrics

- MAE on training and test sets  
- % of predictions exactly correct  
- % within ±1, ±2, ±3 laps  
- Comparison to naive baseline using mean prediction

---

### Overfitting Mitigation

- Race-level splitting prevents leakage across laps in the same event  
- Feature pruning removes indirect race identifiers to boost generalization  
- Early stopping and regularization parameters in LightGBM reduce model complexity  
- Model performance is monitored across train/test splits to ensure minimal gap (Train MAE ≈ 3.455 vs. Test MAE ≈ 3.900)

---

## Sample Visualization

![Sample Model Output](https://github.com/k-dickinson/f1-pit-strat-model/blob/main/F1_sample_output.png)

- Left: Feature importance from LightGBM  
- Right: Evaluation summary including MAE and lap-level prediction accuracy

code available [here!](https://github.com/k-dickinson/f1-pit-strat-model/blob/main/main.py)
---

## Future Improvements

- Integrate metadata per circuit: pit delta, degradation curves, surface type  
- Expand dataset to include multiple seasons for robustness  
- Explore time-aware models (e.g. LSTM, Transformers)  
- Build real-time simulation interface for strategy overlays

---

## Why It Matters

F1 pit strategy prediction is a highly complex task influenced by tire behavior, safety car triggers, driver style, and track dynamics.  
This project demonstrates that even with minimal inputs, machine learning can anticipate strategic behavior with surprising accuracy.

---

## Contact

- Built by [@k-dickinson](https://github.com/k-dickinson)  
- Instagram: [@Quant_Kyle](https://instagram.com/quant_kyle)  
- Feedback, collaborations, and race insights welcome!
