import pandas as pd
from sklearn.metrics import fairness_metrics


def detect_bias(dataset, sensitive_attribute, target_attribute):
    # ... existing code ...
    fairness_report = fairness_metrics(dataset, sensitive_attribute, target_attribute)
    return fairness_report
