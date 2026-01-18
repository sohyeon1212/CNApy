try:
    import gurobipy
except ImportError:
    gurobipy = None
import io
import traceback

import cobra
from qtpy.QtWidgets import QMessageBox


def except_likely_community_model_error() -> None:
    """Shows a message in the case that using a (size-limited) community edition solver version probably caused an error."""
    community_error_text = (
        "Solver error. One possible reason: You set CPLEX or Gurobi as solver although you only use their\n"
        + "Community edition which only work for small models. To solve this, either follow the instructions under\n"
        + "'Config->Configure IBM CPLEX full version' or 'Config->Configure Gurobi full version', or use a different solver such as GLPK."
    )
    msgBox = QMessageBox()
    msgBox.setWindowTitle("Error")
    msgBox.setText(community_error_text)
    msgBox.setIcon(QMessageBox.Warning)
    msgBox.exec()


def get_last_exception_string() -> str:
    output = io.StringIO()
    traceback.print_exc(file=output)
    return output.getvalue()


def has_community_error_substring(string: str) -> bool:
    is_community_error = ("Model too large for size-limited license" in string) or ("1016: Community Edition" in string)

    # Check for Gurobi license errors (codes 10012, 10013 or user reported 12, 13)
    if not is_community_error and ("Gurobi" in string or "gurobi" in string):
        # 10012: Model too large (already covered by "Model too large..." often, but checking code is safer)
        # 10013: Cloud license error?
        # User explicitly mentioned "license 12" and "license 13" so we check for those patterns too.
        is_community_error = any(
            x in string for x in ["10012", "10013", "license 12", "license 13", "status 12", "status 13"]
        )

    return is_community_error


def model_optimization_with_exceptions(model: cobra.Model) -> None:
    try:
        return model.optimize()
    except Exception:
        exstr = get_last_exception_string()
        # Check for substrings of Gurobi and CPLEX community edition errors
        if has_community_error_substring(exstr):
            except_likely_community_model_error()
