"""
test_demo_dataset.py
---------------------
Tests the ready-to-upload demo dataset files in demo_dataset/ to ensure
they cleanly pass through Cyber Guardian AI's CSV loading, schema validation,
and end-to-end Random Forest prediction workflows.
"""

import os
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from src.data_processing import (
    REQUIRED_FEATURE_COLUMNS,
    generate_synthetic_dataset,
    load_csv_bytes,
    preprocess_dataframe,
    validate_dataframe,
)
from src.detection import get_threat_summary, run_prediction, run_training


DEMO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "demo_dataset")
UNLABELLED_CSV = os.path.join(DEMO_DIR, "cyber_guardian_demo.csv")
LABELLED_CSV = os.path.join(DEMO_DIR, "cyber_guardian_demo_labelled.csv")
README_TXT = os.path.join(DEMO_DIR, "README.txt")


def test_demo_dataset_files_exist():
    """Verify all required files exist in demo_dataset/."""
    assert os.path.isdir(DEMO_DIR), f"Missing demo directory: {DEMO_DIR}"
    assert os.path.isfile(UNLABELLED_CSV), f"Missing {UNLABELLED_CSV}"
    assert os.path.isfile(LABELLED_CSV), f"Missing {LABELLED_CSV}"
    assert os.path.isfile(README_TXT), f"Missing {README_TXT}"


def test_unlabelled_csv_schema_and_size():
    """Verify cyber_guardian_demo.csv has exactly 1,000 records and 20 features without label."""
    df = pd.read_csv(UNLABELLED_CSV)
    assert len(df) == 1000
    assert list(df.columns) == REQUIRED_FEATURE_COLUMNS
    assert "label" not in df.columns


def test_labelled_csv_schema_and_distribution():
    """Verify cyber_guardian_demo_labelled.csv has 1,000 records, 21 columns, and 5 classes."""
    df = pd.read_csv(LABELLED_CSV)
    assert len(df) == 1000
    assert list(df.columns) == REQUIRED_FEATURE_COLUMNS + ["label"]
    counts = df["label"].value_counts().to_dict()
    assert counts.get("BENIGN") == 600
    assert counts.get("DoS") == 100
    assert counts.get("DDoS") == 100
    assert counts.get("PortScan") == 100
    assert counts.get("BruteForce") == 100


def test_upload_and_prediction_pipeline():
    """Simulate user upload and end-to-end prediction execution."""
    with open(UNLABELLED_CSV, "rb") as f:
        file_bytes = f.read()

    # 1. Simulates Streamlit upload load_csv_bytes
    loaded_df, err = load_csv_bytes(file_bytes)
    assert err is None
    assert loaded_df is not None

    # 2. Schema validation
    val_df, val_err = validate_dataframe(loaded_df, require_label=False)
    assert val_err is None
    assert val_df is not None
    assert len(val_df) == 1000

    # 3. Baseline training (as done on startup in app.py)
    baseline_df = generate_synthetic_dataset(n_benign=600, n_attack=400, random_state=42)
    tr = run_training(baseline_df)
    assert tr.success is True
    assert tr.pipeline is not None

    # 4. End-to-end prediction
    pred_res = run_prediction(tr.pipeline, val_df)
    assert pred_res.success is True
    assert pred_res.result_df is not None
    assert len(pred_res.result_df) == 1000
    assert "predicted_label" in pred_res.result_df.columns
    assert "confidence_score" in pred_res.result_df.columns
    assert "risk_level" in pred_res.result_df.columns
    assert "threat_category" in pred_res.result_df.columns
    assert "explanation" in pred_res.result_df.columns
    assert "recommended_action" in pred_res.result_df.columns

    # 5. Downstream threat summary
    summary = get_threat_summary(pred_res.result_df)
    assert summary["total"] == 1000
    assert summary["benign_count"] > 0
    assert summary["attack_count"] > 0
    assert summary["high_risk"] > 0

    # 6. Empirical accuracy check against labelled ground truth
    df_labelled = pd.read_csv(LABELLED_CSV)
    acc = accuracy_score(df_labelled["label"], pred_res.predictions)
    assert acc >= 0.85, f"Expected empirical accuracy >= 85%, got {acc * 100:.2f}%"
