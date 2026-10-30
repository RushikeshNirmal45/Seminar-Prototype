"""
test_analyzer_and_triage.py
---------------------------
Tests for enhanced network traffic analysis, triage metadata,
and visualization functions.
"""

import io
import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    LABEL_COLUMN,
    REQUIRED_FEATURE_COLUMNS,
    detect_suspicious_patterns,
    generate_sample_csv,
    generate_synthetic_dataset,
    get_numeric_correlation,
)
from src.detection import (
    THREAT_METADATA,
    get_threat_metadata,
    run_prediction,
    run_training,
)
from src.visualizations import (
    plot_correlation_heatmap,
    plot_protocol_distribution,
    plot_threat_category_breakdown,
    plot_traffic_volume,
)


class TestTrafficAnalyzerHelpers:

    def test_get_numeric_correlation(self):
        df = generate_synthetic_dataset(n_benign=30, n_attack=20)
        corr = get_numeric_correlation(df)
        assert isinstance(corr, pd.DataFrame)
        assert not corr.empty
        assert "duration" in corr.columns
        assert "count" in corr.columns
        # Correlation with self should be 1.0
        assert np.isclose(corr.loc["duration", "duration"], 1.0)

    def test_get_numeric_correlation_empty_df(self):
        empty_df = pd.DataFrame()
        corr = get_numeric_correlation(empty_df)
        assert corr.empty

    def test_detect_suspicious_patterns(self):
        df = generate_synthetic_dataset(n_benign=50, n_attack=50)
        analyzed = detect_suspicious_patterns(df)
        assert "suspicion_reasons" in analyzed.columns
        assert len(analyzed) == len(df)
        # Should detect non-normal patterns in attack rows
        flagged = analyzed[analyzed["suspicion_reasons"] != "Normal"]
        assert len(flagged) > 0

    def test_detect_suspicious_patterns_empty(self):
        res = detect_suspicious_patterns(pd.DataFrame())
        assert res.empty

    def test_generate_sample_csv(self):
        raw_labeled = generate_sample_csv(has_label=True, n_records=25)
        assert isinstance(raw_labeled, bytes)
        df = pd.read_csv(io.BytesIO(raw_labeled))
        assert len(df) == 25
        assert LABEL_COLUMN in df.columns

        raw_unlabeled = generate_sample_csv(has_label=False, n_records=25)
        df_unl = pd.read_csv(io.BytesIO(raw_unlabeled))
        assert len(df_unl) == 25
        assert LABEL_COLUMN not in df_unl.columns

        # Verify n_rows alias works interchangeably
        raw_alias = generate_sample_csv(n_rows=15)
        df_alias = pd.read_csv(io.BytesIO(raw_alias))
        assert len(df_alias) == 15


class TestThreatTriageAndMetadata:

    def test_threat_metadata_supported_classes(self):
        classes = ["BENIGN", "DoS", "DDoS", "PortScan", "BruteForce"]
        for cls in classes:
            meta = get_threat_metadata(cls)
            assert "category" in meta
            assert "severity" in meta
            assert "description" in meta
            assert "indicators" in meta
            assert "impact" in meta
            assert "recommended_action" in meta

    def test_threat_metadata_unknown_class(self):
        meta = get_threat_metadata("UnknownExploit")
        assert "category" in meta
        assert "recommended_action" in meta

    def test_enriched_prediction_columns(self):
        train_df = generate_synthetic_dataset(n_benign=60, n_attack=40, random_state=42)
        train_res = run_training(train_df)
        assert train_res.success

        test_data = train_df.drop(columns=[LABEL_COLUMN]).head(20)
        pred_res = run_prediction(train_res.pipeline, test_data)
        assert pred_res.success
        assert "predicted_label" in pred_res.result_df.columns
        assert "confidence_score" in pred_res.result_df.columns
        assert "risk_level" in pred_res.result_df.columns
        assert "threat_category" in pred_res.result_df.columns
        assert "explanation" in pred_res.result_df.columns
        assert "recommended_action" in pred_res.result_df.columns


class TestNewVisualizations:

    def test_plot_correlation_heatmap(self):
        df = generate_synthetic_dataset(n_benign=20, n_attack=20)
        corr = get_numeric_correlation(df)
        fig = plot_correlation_heatmap(corr)
        assert fig is not None
        assert hasattr(fig, "data")

    def test_plot_protocol_distribution(self):
        df = generate_synthetic_dataset(n_benign=20, n_attack=20)
        fig = plot_protocol_distribution(df)
        assert fig is not None

    def test_plot_traffic_volume(self):
        df = generate_synthetic_dataset(n_benign=20, n_attack=20)
        fig = plot_traffic_volume(df)
        assert fig is not None

    def test_plot_threat_category_breakdown(self):
        df = pd.DataFrame({
            "threat_category": ["Normal Network Traffic", "Reconnaissance / Network Probing", "Normal Network Traffic"],
        })
        fig = plot_threat_category_breakdown(df)
        assert fig is not None

