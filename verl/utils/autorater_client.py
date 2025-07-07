import requests
from typing import Any, Dict, List, Tuple

__all__ = ["call_autorater_service"]

def call_autorater_service(
    base_url: str,
    payload: Dict[str, Any],
    batch_size: int,
    endpoint: str = "/evaluate_autorater",
    timeout: int = 600,
) -> Tuple[List[float], List[int], List[str], List[str]]:
    """Call the remote AutoRater service and return shaped scores.

    Args:
        base_url: Base URL of the AutoRater service (e.g. "http://127.0.0.1:8000").
        payload: JSON payload following the AutoRaterRequest schema.
        batch_size: Expected batch size – used for fallback defaults.
        endpoint: Endpoint path to call (default: "/evaluate_autorater").
        timeout: Request timeout (seconds).

    Returns:
        Tuple containing (scores, decisions, explanations, raw_responses)
    """

    full_url = f"{base_url.rstrip('/')}{endpoint}"
    response = requests.post(full_url, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    scores = data.get("autorater_scores", [0.0] * batch_size)
    decisions = data.get("autorater_decisions", [-1] * batch_size)
    explanations = data.get("autorater_explanations", ["N/A"] * batch_size)
    raw = data.get("autorater_raw_responses", ["N/A"] * batch_size)

    # Shape scores using decision labels to keep backward-compat behaviour
    shaped_scores: List[float] = []
    for dec, raw_score in zip(decisions, scores):
        if dec == 1:
            shaped_scores.append(2.0)
        elif dec == 0:
            shaped_scores.append(-1.5)
        else:
            shaped_scores.append(raw_score)

    return shaped_scores, decisions, explanations, raw 