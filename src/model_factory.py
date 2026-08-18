from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier


def get_models():
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss"
        ),
        "mlp_shallow": MLPClassifier(
            hidden_layer_sizes=(32,),
            max_iter=500,
            random_state=42
        ),
        "mlp_deep": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            random_state=42
        )
    }