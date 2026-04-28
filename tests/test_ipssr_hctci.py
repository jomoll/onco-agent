import json

import pytest

from src.agent_tools import HCTCITool, IPSSRScoreTool


def _parse_json_output(tool_output):
    return json.loads(tool_output.content)


class TestIPSSRScoreTool:
    def setup_method(self):
        self.tool = IPSSRScoreTool()

    def test_best_case_very_low(self):
        out = self.tool(
            hemoglobin_g_per_dl=10.0,
            platelets_10e9_per_L=100.0,
            anc_10e9_per_L=0.8,
            marrow_blast_percent=2.0,
            cytogenetic_risk="Very Good",
        )
        data = _parse_json_output(out)
        assert out.is_error is False
        assert data["total_score"] == pytest.approx(0.0)
        assert data["risk_category"] == "Very Low"

    def test_high_score_example(self):
        out = self.tool(
            hemoglobin_g_per_dl=7.9,  # 1.5
            platelets_10e9_per_L=49,  # 1
            anc_10e9_per_L=0.79,  # 0.5
            marrow_blast_percent=4.9,  # 1
            cytogenetic_risk="Good",  # 1
        )
        data = _parse_json_output(out)
        assert data["total_score"] == pytest.approx(5.0)
        assert data["risk_category"] == "High"

    @pytest.mark.parametrize(
        "score_kwargs,expected_risk,expected_total",
        [
            (
                dict(
                    hemoglobin_g_per_dl=7.9,  # 1.5
                    platelets_10e9_per_L=200,
                    anc_10e9_per_L=2.0,
                    marrow_blast_percent=1.0,
                    cytogenetic_risk="Very Good",
                ),
                "Very Low",
                1.5,
            ),
            (
                dict(
                    hemoglobin_g_per_dl=9.0,  # 1
                    platelets_10e9_per_L=200,
                    anc_10e9_per_L=2.0,
                    marrow_blast_percent=4.0,  # 1
                    cytogenetic_risk="Good",  # 1
                ),
                "Low",
                3.0,
            ),
            (
                dict(
                    hemoglobin_g_per_dl=7.9,  # 1.5
                    platelets_10e9_per_L=200,
                    anc_10e9_per_L=2.0,
                    marrow_blast_percent=2.0,  # 0
                    cytogenetic_risk="Poor",  # 3
                ),
                "Intermediate",
                4.5,
            ),
            (
                dict(
                    hemoglobin_g_per_dl=9.0,  # 1
                    platelets_10e9_per_L=75.0,  # 0.5
                    anc_10e9_per_L=0.7,  # 0.5
                    marrow_blast_percent=6.0,  # 2
                    cytogenetic_risk="Very Poor",  # 4
                ),
                "Very High",
                8.0,
            ),
        ],
    )
    def test_category_boundaries(self, score_kwargs, expected_risk, expected_total):
        out = self.tool(**score_kwargs)
        data = _parse_json_output(out)
        assert data["risk_category"] == expected_risk
        assert data["total_score"] == pytest.approx(expected_total)

    def test_missing_parameter_yields_error(self):
        out = self.tool(
            hemoglobin_g_per_dl=10.0,
            platelets_10e9_per_L=None,
            anc_10e9_per_L=1.0,
            marrow_blast_percent=1.0,
            cytogenetic_risk="Good",
        )
        assert out.is_error is True
        assert "Missing required parameters" in out.content


class TestHCTCITool:
    BASE_FLAGS = {
        "arrhythmia": False,
        "cardiac_disease": False,
        "inflammatory_bowel_disease": False,
        "diabetes_medication": False,
        "cerebrovascular_disease": False,
        "psychiatric_disturbance": False,
        "mild_hepatic_abnormality": False,
        "obesity_bmi_gt_35": False,
        "persistent_infection": False,
        "rheumatologic_disease": False,
        "peptic_ulcer": False,
        "renal_moderate_severe": False,
        "pulmonary_moderate": False,
        "prior_solid_tumor": False,
        "heart_valve_disease": False,
        "pulmonary_severe": False,
        "hepatic_moderate_severe": False,
    }

    def setup_method(self):
        self.tool = HCTCITool()

    def _call(self, **overrides):
        flags = dict(self.BASE_FLAGS)
        flags.update(overrides)
        return self.tool(**flags)

    def test_low_risk_zero_score(self):
        out = self._call()
        data = _parse_json_output(out)
        assert out.is_error is False
        assert data["total_score"] == 0
        assert data["risk_group"] == "low"

    def test_intermediate_risk(self):
        out = self._call(arrhythmia=True, obesity_bmi_gt_35=True)
        data = _parse_json_output(out)
        assert data["total_score"] == 2
        assert data["risk_group"] == "intermediate"

    def test_high_risk(self):
        out = self._call(rheumatologic_disease=True, prior_solid_tumor=True)
        data = _parse_json_output(out)
        assert data["total_score"] == 5
        assert data["risk_group"] == "high"

    def test_missing_flags_error(self):
        out = self.tool(
            arrhythmia=True,
            cardiac_disease=True,
            inflammatory_bowel_disease=True,
            diabetes_medication=True,
            cerebrovascular_disease=True,
            psychiatric_disturbance=True,
            mild_hepatic_abnormality=True,
            obesity_bmi_gt_35=True,
            persistent_infection=True,
            rheumatologic_disease=True,
            peptic_ulcer=True,
            renal_moderate_severe=True,
            pulmonary_moderate=True,
            prior_solid_tumor=None,  # intentionally missing
            heart_valve_disease=True,
            pulmonary_severe=True,
            hepatic_moderate_severe=True,
        )
        assert out.is_error is True
        assert "Missing required comorbidity flags" in out.content
