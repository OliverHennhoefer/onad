from __future__ import annotations

import numpy as np

from scipy import integrate

from onad.base.metric import BaseMetric


class ROCAUC(BaseMetric):
    def __init__(self, n_thresholds=10, pos_val=True):
        if n_thresholds < 2:
            raise ValueError("n_thresholds must be greater than 1.")
        self.n_thresholds = n_thresholds
        self.pos_val = pos_val
        self.thresholds = [i / (n_thresholds - 1) for i in range(n_thresholds)]
        self.thresholds[0] -= 1e-7
        self.thresholds[-1] += 1e-7
        # Initialize confusion matrices for each threshold
        self.cms = [{"tp": 0, "tn": 0, "fp": 0, "fn": 0} for _ in range(n_thresholds)]

    import numpy as np  # Add this import at the top of your file

    def update(self, y_true, y_pred, w=1.0):
        # Skip update if y_pred is None (e.g., during warm-up)
        if y_pred is None:
            return

        # Extract p_true from y_pred
        if isinstance(y_pred, dict):
            p_true = y_pred.get(True, 0.0)
        else:
            p_true = y_pred

        # Ensure p_true is a valid number (including NumPy numeric types)
        if not isinstance(p_true, (int, float, np.number)):
            raise TypeError(
                f"y_pred must be a number or a dictionary with a valid key. Got: {type(p_true)}"
            )

        # Update confusion matrices for each threshold
        for t, cm in zip(self.thresholds, self.cms):
            pred_pos = p_true > t
            actual_pos = y_true == self.pos_val
            if actual_pos and pred_pos:
                cm["tp"] += w
            elif actual_pos and not pred_pos:
                cm["fn"] += w
            elif not actual_pos and pred_pos:
                cm["fp"] += w
            else:
                cm["tn"] += w

    def get(self):
        tprs = [0] * self.n_thresholds
        fprs = [0] * self.n_thresholds

        def safe_div(a, b):
            try:
                return a / b
            except ZeroDivisionError:
                return 0.0

        for i, cm in enumerate(self.cms):
            tp = cm["tp"]
            tn = cm["tn"]
            fp = cm["fp"]
            fn = cm["fn"]

            tprs[i] = safe_div(a=tp, b=tp + fn)  # True Positive Rate (Recall)
            fprs[i] = safe_div(a=fp, b=fp + tn)  # False Positive Rate

        return -integrate.trapezoid(x=fprs, y=tprs)
