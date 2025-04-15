import shap
from sklearn.ensemble import RandomForestClassifier


def explain_model(model, data):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data)
    return shap_values
