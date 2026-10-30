"""
detection.py
------------
Orchestrates the end-to-end detection workflow:
  1. Preprocess raw data.
  2. Train the model (if labelled data is provided).
  3. Predict on new records.
  4. Evaluate against ground truth (if available).

This module acts as the glue between data_processing, model, and evaluation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data_processing import (
    LABEL_COLUMN,
    REQUIRED_FEATURE_COLUMNS,
    get_class_distribution,
    get_dataset_summary,
    preprocess_dataframe,
)
from src.evaluation import compute_metrics, format_per_class_metrics
from src.model import get_feature_importance, get_model_info, predict, train_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Threat Knowledge Base & Triage Metadata
# ---------------------------------------------------------------------------

THREAT_METADATA: Dict[str, Dict[str, str]] = {
    "BENIGN": {
        "category": "Normal Network Traffic",
        "severity": "LOW",
        "description": "Standard network communication exhibiting normal bidirectional byte flow, typical duration, and service distribution without flood or intrusion patterns.",
        "indicators": "Normal duration (>1s), balanced source/destination bytes, low connection counts (<50), zero failed logins.",
        "impact": "None. Legitimate operational network traffic.",
        "recommended_action": "No immediate defensive action required. Maintain standard baseline telemetry logging and monitor for unusual behavioral shifts.",
    },
    "DoS": {
        "category": "Denial of Service (Volume/State Exhaustion)",
        "severity": "HIGH",
        "description": "High-volume connection flood originating from a single or concentrated source designed to exhaust server memory, connection tables, or CPU.",
        "indicators": "Extremely high connection count (>200), high same-service rate (>0.85), short flow duration (<0.5s), small byte payloads.",
        "impact": "Service degradation, connection timeouts, daemon crashes, unresponsiveness for legitimate users.",
        "recommended_action": "Implement rate limiting at edge firewall/router, enable TCP SYN cookies, block offending source IP address, and inspect edge connection limits.",
    },
    "DDoS": {
        "category": "Distributed Denial of Service",
        "severity": "CRITICAL",
        "description": "Coordinated flood from multiple distributed endpoints targeting network infrastructure or application ports to cause widespread service unavailability.",
        "indicators": "Massive influx of connections, concentrated dst_host_count, near-identical service requests across distributed sessions.",
        "impact": "Complete service outage, upstream transit link saturation, firewall state-table exhaustion, collateral service impairment.",
        "recommended_action": "Divert traffic through upstream BGP Anycast scrubbing center / CDN DDoS mitigation, engage ISP DDoS protection, enforce edge geo/rate-limiting.",
    },
    "PortScan": {
        "category": "Reconnaissance / Network Probing",
        "severity": "MEDIUM",
        "description": "Systematic scanning of host IP addresses and ports to map active listening services, vulnerable daemons, and network topology prior to exploitation.",
        "indicators": "Rapid connections with high diff_srv_rate (>0.7), low srv_count (<5), very low dwell time per port, same source port reuse.",
        "impact": "Exposure of open ports, daemon versions, and perimeter defense gaps to adversary reconnaissance.",
        "recommended_action": "Temporarily block or quarantine scanning source IP, close non-essential listening ports, verify firewall ACL rules, and alert SOC tier-1 analyst.",
    },
    "BruteForce": {
        "category": "Credential Access / Authentication Attack",
        "severity": "HIGH",
        "description": "Automated, repeated authentication attempts against network login services (SSH, FTP, RDP, Web) to guess usernames and passwords.",
        "indicators": "Repeated connection sessions to authentication ports, elevated num_failed_logins (>=3), high hot count, repeated access attempts.",
        "impact": "Unauthorized account compromise, privilege escalation, persistent access, internal network pivot.",
        "recommended_action": "Enforce account lockout / rate-limiting, terminate active compromised sessions, enforce Multi-Factor Authentication (MFA), deploy Fail2Ban.",
    },
}


def get_threat_metadata(label: str) -> Dict[str, str]:
    """Return threat metadata dictionary for a given label."""
    return THREAT_METADATA.get(str(label), {
        "category": "Unclassified Traffic",
        "severity": "MEDIUM",
        "description": f"Traffic pattern classified as {label}.",
        "indicators": "Statistical divergence from normal baseline flows.",
        "impact": "Requires SOC analyst investigation.",
        "recommended_action": "Inspect packet headers and protocol telemetry for unusual anomalies.",
    })


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Holds the outcome of a model training run."""
    success: bool
    pipeline: Optional[Pipeline] = None
    X_test: Optional[pd.DataFrame] = None
    y_test: Optional[pd.Series] = None
    train_samples: int = 0
    test_samples: int = 0
    classes: List[str] = field(default_factory=list)
    error: Optional[str] = None
    model_info: Dict = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Holds the outcome of a prediction run."""
    success: bool
    predictions: Optional[np.ndarray] = None
    probabilities: Optional[np.ndarray] = None
    result_df: Optional[pd.DataFrame] = None
    error: Optional[str] = None


@dataclass
class EvaluationResult:
    """Holds model evaluation metrics."""
    success: bool
    metrics: Optional[Dict] = None
    per_class_df: Optional[pd.DataFrame] = None
    feature_importance: Optional[pd.DataFrame] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Detection Workflow
# ---------------------------------------------------------------------------

def run_training(df: pd.DataFrame) -> TrainingResult:
    """
    Preprocess the labelled dataset and train the classifier.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (may be from CSV upload or synthetic generation).

    Returns
    -------
    TrainingResult
    """
    cleaned_df, err = preprocess_dataframe(df, require_label=True)
    if err:
        return TrainingResult(success=False, error=err)

    pipeline, X_test, y_test, err = train_model(cleaned_df)
    if err:
        return TrainingResult(success=False, error=err)

    n_train = len(cleaned_df) - len(X_test)
    classes = list(cleaned_df[LABEL_COLUMN].unique())

    return TrainingResult(
        success=True,
        pipeline=pipeline,
        X_test=X_test,
        y_test=y_test,
        train_samples=n_train,
        test_samples=len(X_test),
        classes=sorted(classes),
        model_info=get_model_info(pipeline),
    )


def run_evaluation(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> EvaluationResult:
    """
    Evaluate a trained pipeline on the held-out test set.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted scikit-learn Pipeline.
    X_test : pd.DataFrame
        Features of the held-out test set.
    y_test : pd.Series
        Ground-truth labels for X_test.

    Returns
    -------
    EvaluationResult
    """
    y_pred, _, err = predict(pipeline, X_test)
    if err:
        return EvaluationResult(success=False, error=err)

    metrics, err = compute_metrics(y_test, y_pred)
    if err:
        return EvaluationResult(success=False, error=err)

    per_class_df = format_per_class_metrics(metrics)
    feature_importance = get_feature_importance(pipeline)

    return EvaluationResult(
        success=True,
        metrics=metrics,
        per_class_df=per_class_df,
        feature_importance=feature_importance,
    )


def run_prediction(
    pipeline: Pipeline,
    df: pd.DataFrame,
    has_labels: bool = False,
) -> PredictionResult:
    """
    Run predictions on a new dataset using a trained pipeline.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted scikit-learn Pipeline.
    df : pd.DataFrame
        Raw feature data (may optionally contain a label column).
    has_labels : bool
        Whether the label column is present (used for display, not training).

    Returns
    -------
    PredictionResult
    """
    cleaned_df, err = preprocess_dataframe(df, require_label=False)
    if err:
        return PredictionResult(success=False, error=err)

    available_features = [c for c in REQUIRED_FEATURE_COLUMNS if c in cleaned_df.columns]
    if not available_features:
        return PredictionResult(
            success=False,
            error="No recognised feature columns found in the uploaded data.",
        )

    X = cleaned_df[available_features]

    y_pred, y_proba, err = predict(pipeline, X)
    if err:
        return PredictionResult(success=False, error=err)

    result_df = cleaned_df.copy()
    result_df["predicted_label"] = y_pred

    # Attach max probability as a risk confidence score
    if y_proba is not None:
        result_df["confidence_score"] = np.max(y_proba, axis=1).round(4)

    # Add a risk flag: BENIGN → Low, else High
    result_df["risk_level"] = result_df["predicted_label"].apply(
        lambda lbl: "LOW" if str(lbl).upper() == "BENIGN" else "HIGH"
    )

    # Attach threat category, explanation, and recommended action
    categories = []
    explanations = []
    actions = []

    for idx, label in enumerate(y_pred):
        meta = get_threat_metadata(str(label))
        categories.append(meta.get("category", "Traffic Flow"))
        actions.append(meta.get("recommended_action", "Review traffic telemetry."))

        row = cleaned_df.iloc[idx]
        lbl_str = str(label).upper()
        if lbl_str == "BENIGN":
            explanations.append("Standard bidirectional telemetry; no flooding, scanning, or authentication anomalies observed.")
        elif lbl_str == "DOS":
            c_val = row.get("count", 0)
            explanations.append(f"Elevated connection density (count={c_val:.0f}) and short dwell time consistent with volume/state exhaustion flood.")
        elif lbl_str == "DDOS":
            dst_host = row.get("dst_host_count", 0)
            explanations.append(f"High destination host concentration (dst_host_count={dst_host:.0f}) characteristic of coordinated distributed flooding.")
        elif lbl_str == "PORTSCAN":
            diff_srv = row.get("diff_srv_rate", 0)
            explanations.append(f"Disproportionate service variation (diff_srv_rate={diff_srv:.2f}) across short-duration connections indicating port reconnaissance.")
        elif lbl_str == "BRUTEFORCE":
            failed = row.get("num_failed_logins", 0)
            explanations.append(f"Repeated authentication challenges detected (num_failed_logins={failed:.0f}) indicating credential brute-forcing.")
        else:
            explanations.append(f"Flagged by classifier based on similarity to known {label} training vector patterns.")

    result_df["threat_category"] = categories
    result_df["explanation"] = explanations
    result_df["recommended_action"] = actions

    return PredictionResult(
        success=True,
        predictions=y_pred,
        probabilities=y_proba,
        result_df=result_df,
    )


def get_threat_summary(result_df: pd.DataFrame) -> Dict:
    """
    Summarise prediction results for the dashboard.

    Returns a dict with counts, proportions, and a colour-coded breakdown.
    """
    if result_df is None or result_df.empty:
        return {}

    total = len(result_df)
    if "predicted_label" not in result_df.columns:
        return {"total": total}

    label_counts = result_df["predicted_label"].value_counts().to_dict()
    benign_count = label_counts.get("BENIGN", 0)
    attack_count = total - benign_count

    return {
        "total": total,
        "benign_count": benign_count,
        "attack_count": attack_count,
        "benign_pct": round(benign_count / total * 100, 1) if total else 0,
        "attack_pct": round(attack_count / total * 100, 1) if total else 0,
        "label_counts": label_counts,
        "high_risk": int((result_df["risk_level"] == "HIGH").sum()),
        "low_risk": int((result_df["risk_level"] == "LOW").sum()),
    }

