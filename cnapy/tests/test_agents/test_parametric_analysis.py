"""Tests for Parametric Analysis Skill

This module tests the parametric analysis skill that enables
requests like "Growth rate 0~100% 10%씩 pFBA".
"""

from cnapy.agents.base_agent import SkillStatus


class TestParametricAnalysisSkillRegistration:
    """Test that parametric_analysis skill is properly registered."""

    def test_skill_registered(self, agent_context):
        """Test that parametric_analysis skill is registered."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context)

        assert "parametric_analysis" in agent.skills

    def test_skill_parameters(self, agent_context):
        """Test parametric_analysis skill parameters."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context)
        skill = agent.skills["parametric_analysis"]

        assert "analysis_type" in skill.parameters
        assert "parameter" in skill.parameters
        assert "start_percent" in skill.parameters
        assert "end_percent" in skill.parameters
        assert "step_percent" in skill.parameters

    def test_skill_required_params(self, agent_context):
        """Test parametric_analysis required parameters."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context)
        skill = agent.skills["parametric_analysis"]

        required = skill.required_params
        assert "analysis_type" in required
        assert "parameter" in required
        assert "start_percent" in required
        assert "end_percent" in required
        assert "step_percent" not in required  # Has default


class TestParametricAnalysisNoModel:
    """Test parametric analysis when no model is loaded."""

    def test_no_model_returns_failure(self, agent_context):
        """Test that parametric analysis fails gracefully without model."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        # Create context with no model
        agent_context.appdata.project.cobra_py_model = None
        agent = FluxAnalysisAgent(agent_context)

        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "pfba",
                "parameter": "growth_rate",
                "start_percent": 0,
                "end_percent": 100,
            },
        )

        assert result.status == SkillStatus.FAILURE
        assert "no" in result.message.lower() or "model" in result.message.lower()


class TestParametricAnalysisWithEcoliModel:
    """Test parametric analysis with E. coli core model."""

    def test_pfba_growth_rate_sweep(self, agent_context_ecoli):
        """Test pFBA with growth rate sweep."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context_ecoli)

        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "pfba",
                "parameter": "growth_rate",
                "start_percent": 0,
                "end_percent": 50,
                "step_percent": 25,
            },
        )

        assert result.status == SkillStatus.SUCCESS
        assert "parametric_results" in result.data
        assert len(result.data["parametric_results"]) == 3  # 0%, 25%, 50%

    def test_fba_growth_rate_sweep(self, agent_context_ecoli):
        """Test FBA with growth rate sweep."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context_ecoli)

        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "fba",
                "parameter": "growth_rate",
                "start_percent": 0,
                "end_percent": 100,
                "step_percent": 50,
            },
        )

        assert result.status == SkillStatus.SUCCESS
        assert "parametric_results" in result.data
        # Should have 3 points: 0%, 50%, 100%
        assert len(result.data["parametric_results"]) == 3

    def test_parametric_results_structure(self, agent_context_ecoli):
        """Test the structure of parametric results."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context_ecoli)

        # Use FBA instead of pFBA to avoid numpy compatibility issues
        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "fba",
                "parameter": "growth_rate",
                "start_percent": 50,
                "end_percent": 50,
                "step_percent": 10,
            },
        )

        assert result.status == SkillStatus.SUCCESS
        data_point = result.data["parametric_results"][0]

        assert "fraction" in data_point
        assert "status" in data_point

    def test_parametric_metadata(self, agent_context_ecoli):
        """Test parametric analysis metadata."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context_ecoli)

        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "pfba",
                "parameter": "growth_rate",
                "start_percent": 0,
                "end_percent": 100,
                "step_percent": 50,
            },
        )

        assert result.status == SkillStatus.SUCCESS
        assert result.data["parameter"] == "growth_rate"
        assert result.data["analysis_type"] == "pfba"
        assert "max_growth" in result.data


class TestParametricAnalysisCancellation:
    """Test cancellation support for parametric analysis."""

    def test_cancel_check_callback(self, agent_context_ecoli):
        """Test that cancel check callback is called."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        cancel_count = [0]

        def cancel_check():
            cancel_count[0] += 1
            return cancel_count[0] > 2  # Cancel after 2 checks

        agent = FluxAnalysisAgent(agent_context_ecoli, cancel_check=cancel_check)

        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "fba",
                "parameter": "growth_rate",
                "start_percent": 0,
                "end_percent": 100,
                "step_percent": 10,
            },
        )

        # Should be cancelled partway through
        assert result.status == SkillStatus.PARTIAL
        assert "cancelled" in result.message.lower()

    def test_set_cancel_check(self, agent_context_ecoli):
        """Test setting cancel check callback after initialization."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context_ecoli)

        cancel_flag = [False]

        def cancel_check():
            return cancel_flag[0]

        agent.set_cancel_check(cancel_check)

        # Start normally
        cancel_flag[0] = False
        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "fba",
                "parameter": "growth_rate",
                "start_percent": 0,
                "end_percent": 20,
                "step_percent": 10,
            },
        )

        assert result.status == SkillStatus.SUCCESS


class TestParametricAnalysisEdgeCases:
    """Test edge cases for parametric analysis."""

    def test_single_point(self, agent_context_ecoli):
        """Test with single point (start == end)."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context_ecoli)

        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "fba",
                "parameter": "growth_rate",
                "start_percent": 50,
                "end_percent": 50,
                "step_percent": 10,
            },
        )

        assert result.status == SkillStatus.SUCCESS
        assert len(result.data["parametric_results"]) == 1

    def test_zero_growth(self, agent_context_ecoli):
        """Test with zero growth rate."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context_ecoli)

        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "fba",
                "parameter": "growth_rate",
                "start_percent": 0,
                "end_percent": 0,
                "step_percent": 10,
            },
        )

        assert result.status == SkillStatus.SUCCESS
        # At 0% growth, should still get a result
        assert len(result.data["parametric_results"]) >= 1

    def test_full_range(self, agent_context_ecoli):
        """Test full 0-100% range with 10% steps."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context_ecoli)

        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "pfba",
                "parameter": "growth_rate",
                "start_percent": 0,
                "end_percent": 100,
                "step_percent": 10,
            },
        )

        assert result.status == SkillStatus.SUCCESS
        # Should have 11 points: 0, 10, 20, ..., 100
        assert len(result.data["parametric_results"]) == 11


class TestParametricAnalysisMessages:
    """Test parametric analysis message formatting."""

    def test_success_message_english(self, agent_context_ecoli):
        """Test English success message."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context_ecoli)
        agent.context.current_language = "en"

        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "pfba",
                "parameter": "growth_rate",
                "start_percent": 0,
                "end_percent": 50,
                "step_percent": 25,
            },
        )

        message = result.get_message("en")
        assert "pfba" in message.lower() or "pFBA" in message

    def test_success_message_korean(self, agent_context_ecoli):
        """Test Korean success message."""
        from cnapy.agents.flux_analysis_agent import FluxAnalysisAgent

        agent = FluxAnalysisAgent(agent_context_ecoli)
        agent.context.current_language = "ko"

        result = agent.execute_skill(
            "parametric_analysis",
            {
                "analysis_type": "pfba",
                "parameter": "growth_rate",
                "start_percent": 0,
                "end_percent": 50,
                "step_percent": 25,
            },
        )

        message_ko = result.message_ko
        assert "완료" in message_ko or "포인트" in message_ko


class TestOrchestratorParametricIntentDetection:
    """Test orchestrator pattern matching for parametric analysis."""

    def test_detect_korean_parametric_pattern(self, agent_context):
        """Test detection of Korean parametric pattern."""
        from cnapy.agents.orchestrator_agent import OrchestratorAgent

        orch = OrchestratorAgent(agent_context)
        intent = orch._analyze_intent("Growth rate 0~100% 10%씩 pFBA")

        assert intent["skill"] == "parametric_analysis"
        assert intent["agent"] == "flux_analysis"

    def test_detect_english_parametric_pattern(self, agent_context):
        """Test detection of English parametric pattern."""
        from cnapy.agents.orchestrator_agent import OrchestratorAgent

        orch = OrchestratorAgent(agent_context)
        intent = orch._analyze_intent("pFBA from 0% to 100% in 10% steps")

        assert intent["skill"] == "parametric_analysis"
        assert intent["agent"] == "flux_analysis"

    def test_extract_parametric_params_korean(self, agent_context):
        """Test parameter extraction for Korean text."""
        from cnapy.agents.orchestrator_agent import OrchestratorAgent

        orch = OrchestratorAgent(agent_context)
        params = orch._extract_parameters("Growth rate 0~100% 10%씩 pFBA", "parametric_analysis")

        assert params["analysis_type"] == "pfba"
        assert params["start_percent"] == 0.0
        assert params["end_percent"] == 100.0
        assert params["step_percent"] == 10.0

    def test_extract_parametric_params_english(self, agent_context):
        """Test parameter extraction for English text."""
        from cnapy.agents.orchestrator_agent import OrchestratorAgent

        orch = OrchestratorAgent(agent_context)
        params = orch._extract_parameters("FBA from 0% to 50% in 5% steps", "parametric_analysis")

        assert params["analysis_type"] == "fba"
        assert params["start_percent"] == 0.0
        assert params["end_percent"] == 50.0
        assert params["step_percent"] == 5.0

    def test_default_values(self, agent_context):
        """Test default parameter values."""
        from cnapy.agents.orchestrator_agent import OrchestratorAgent

        orch = OrchestratorAgent(agent_context)
        params = orch._extract_parameters("parametric FBA", "parametric_analysis")

        assert params["parameter"] == "growth_rate"
        assert params["start_percent"] == 0.0
        assert params["end_percent"] == 100.0
        assert params["step_percent"] == 10.0

    def test_distinguish_simple_fba(self, agent_context):
        """Test that simple FBA is not detected as parametric."""
        from cnapy.agents.orchestrator_agent import OrchestratorAgent

        orch = OrchestratorAgent(agent_context)
        intent = orch._analyze_intent("perform FBA")

        assert intent["skill"] == "perform_fba"
        assert intent["agent"] == "flux_analysis"
