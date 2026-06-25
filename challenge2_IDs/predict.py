import importlib
from pathlib import Path

import joblib
import torch

from base_model import BaseModel
from model import ModelArchitecture


def load_model_architecture():
    """
    Load ModelArchitecture from the model.py file located in the same folder
    as this predict.py file.
    """
    current_dir = Path(__file__).resolve().parent
    model_path = current_dir / "model.py"

    spec = importlib.util.spec_from_file_location("student_model", model_path)

    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import model.py from {model_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "ModelArchitecture"):
        raise AttributeError("model.py must define a class named ModelArchitecture")

    return module.ModelArchitecture


class Model(BaseModel):
    """
    Grader-facing prediction wrapper.

    Students usually should not change this file.

    The evaluator will do:

        model = Model()
        model.load("weights.joblib")
        predictions = model.predict(x)

    The predict method must return class indices, not probabilities/logits.
    """

    def __init__(self):
        self.net = ModelArchitecture(num_classes=20)

    def load(self, weights_path: str) -> None:
        state_dict = joblib.load(weights_path)
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            logits = self.net(x)
            preds = logits.argmax(dim=1)
        return preds