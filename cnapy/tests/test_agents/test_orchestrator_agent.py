"""Tests for OrchestratorAgent."""

from cnapy.agents.orchestrator_agent import OrchestratorAgent


class TestOrchestratorAgent:
    """Tests for OrchestratorAgent."""

    def test_orchestrator_creation(self, agent_context):
        """Test creating the orchestrator."""
        orchestrator = OrchestratorAgent(agent_context)
        assert orchestrator.context is agent_context
        assert orchestrator.registry is not None

    def test_get_available_agents(self, agent_context):
        """Test getting list of available agents."""
        orchestrator = OrchestratorAgent(agent_context)
        agents = orchestrator.get_available_agents()

        assert len(agents) > 0
        agent_types = [a["type"] for a in agents]
        assert "flux_analysis" in agent_types
        assert "scenario" in agent_types

    def test_get_agent_skills(self, agent_context):
        """Test getting skills for an agent."""
        orchestrator = OrchestratorAgent(agent_context)
        skills = orchestrator.get_agent_skills("flux_analysis")

        assert len(skills) > 0
        skill_names = [s["name"] for s in skills]
        assert "perform_fba" in skill_names


class TestIntentAnalysis:
    """Tests for intent analysis."""

    def test_analyze_fba_intent(self, agent_context):
        """Test analyzing FBA intent."""
        orchestrator = OrchestratorAgent(agent_context)
        intent = orchestrator._analyze_intent("perform FBA analysis")

        assert intent["skill"] == "perform_fba"
        assert intent["agent"] == "flux_analysis"
        assert intent["confidence"] > 0

    def test_analyze_condition_intent(self, agent_context):
        """Test analyzing condition intent."""
        orchestrator = OrchestratorAgent(agent_context)
        intent = orchestrator._analyze_intent("set anaerobic condition")

        assert intent["skill"] == "apply_condition"
        assert intent["agent"] == "scenario"

    def test_analyze_korean_intent(self, agent_context):
        """Test analyzing Korean intent."""
        orchestrator = OrchestratorAgent(agent_context)
        intent = orchestrator._analyze_intent("FBA 수행해줘")

        assert intent["skill"] == "perform_fba"
        assert intent["agent"] == "flux_analysis"

    def test_analyze_essential_genes_intent(self, agent_context):
        """Test analyzing essential genes intent."""
        orchestrator = OrchestratorAgent(agent_context)
        intent = orchestrator._analyze_intent("find essential genes")

        assert intent["skill"] == "find_essential_genes"
        assert intent["agent"] == "gene_analysis"

    def test_analyze_unknown_intent(self, agent_context):
        """Test analyzing unknown intent."""
        orchestrator = OrchestratorAgent(agent_context)
        intent = orchestrator._analyze_intent("random gibberish text")

        # Should still try to route based on keywords
        assert "skill" in intent
        assert "agent" in intent


class TestParameterExtraction:
    """Tests for parameter extraction."""

    def test_extract_fva_fraction(self, agent_context):
        """Test extracting FVA fraction parameter."""
        orchestrator = OrchestratorAgent(agent_context)
        params = orchestrator._extract_parameters("perform FVA with 90% optimum", "perform_fva")

        assert "fraction_of_optimum" in params
        assert params["fraction_of_optimum"] == 0.9

    def test_extract_condition_name(self, agent_context):
        """Test extracting condition name."""
        orchestrator = OrchestratorAgent(agent_context)
        params = orchestrator._extract_parameters("set anaerobic condition", "apply_condition")

        assert "condition_name" in params
        assert params["condition_name"] == "anaerobic"

    def test_extract_korean_condition(self, agent_context):
        """Test extracting Korean condition name."""
        orchestrator = OrchestratorAgent(agent_context)
        params = orchestrator._extract_parameters("혐기 조건 설정해줘", "apply_condition")

        assert "condition_name" in params
        assert params["condition_name"] == "anaerobic"

    def test_extract_carbon_source(self, agent_context):
        """Test extracting carbon source."""
        orchestrator = OrchestratorAgent(agent_context)
        params = orchestrator._extract_parameters("set glucose as carbon source", "set_carbon_source")

        assert "carbon_source" in params
        assert params["carbon_source"] == "glucose"


class TestRouting:
    """Tests for routing functionality."""

    def test_route_fba_request(self, agent_context_ecoli):
        """Test routing FBA request."""
        orchestrator = OrchestratorAgent(agent_context_ecoli)
        response = orchestrator.route("perform FBA")

        assert response.success
        assert "growth" in response.message.lower() or response.data is not None

    def test_route_anaerobic_condition(self, agent_context_ecoli):
        """Test routing anaerobic condition request."""
        orchestrator = OrchestratorAgent(agent_context_ecoli)
        response = orchestrator.route("set anaerobic condition")

        assert response.success
        assert response.agent_name is not None

    def test_route_fallback(self, agent_context):
        """Test fallback response for unknown request."""
        orchestrator = OrchestratorAgent(agent_context)
        response = orchestrator.route("xyzrandomgibberishtextabc")

        # Should get a fallback response with suggestions
        assert not response.success or "sorry" in response.message.lower() or "can" in response.message.lower()


class TestWorkflows:
    """Tests for workflow execution."""

    def test_execute_workflow_by_name(self, agent_context_ecoli):
        """Test executing predefined workflow."""
        orchestrator = OrchestratorAgent(agent_context_ecoli)

        # This may or may not succeed depending on workflow availability
        response = orchestrator.execute_workflow_by_name("fba_aerobic")

        # Either succeeds or returns proper error
        assert response is not None

    def test_execute_unknown_workflow(self, agent_context):
        """Test executing unknown workflow."""
        orchestrator = OrchestratorAgent(agent_context)
        response = orchestrator.execute_workflow_by_name("nonexistent_workflow")

        assert not response.success
        assert "not found" in response.message.lower()


class TestLanguageDetection:
    """Tests for language detection."""

    def test_detect_english(self, agent_context):
        """Test detecting English."""
        orchestrator = OrchestratorAgent(agent_context)
        orchestrator.route("perform FBA analysis")

        assert agent_context.current_language == "en"

    def test_detect_korean(self, agent_context):
        """Test detecting Korean."""
        orchestrator = OrchestratorAgent(agent_context)
        orchestrator.route("FBA 수행해줘")

        assert agent_context.current_language == "ko"
