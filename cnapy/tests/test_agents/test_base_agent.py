"""Tests for base agent classes."""

import pytest
from cnapy.agents.base_agent import (
    BaseAgent, AgentContext, SkillResult, Skill, SkillStatus,
    ToolDefinition, AgentResponse, WorkflowStep,
)


class TestSkillResult:
    """Tests for SkillResult class."""

    def test_success_result(self):
        """Test creating a successful result."""
        result = SkillResult(
            status=SkillStatus.SUCCESS,
            message="Success message",
            message_ko="성공 메시지",
            data={"key": "value"}
        )
        assert result.success is True
        assert result.get_message("en") == "Success message"
        assert result.get_message("ko") == "성공 메시지"
        assert result.data == {"key": "value"}

    def test_failure_result(self):
        """Test creating a failure result."""
        result = SkillResult(
            status=SkillStatus.FAILURE,
            message="Failure message",
            error="Some error occurred"
        )
        assert result.success is False
        assert result.error == "Some error occurred"

    def test_partial_result(self):
        """Test creating a partial result."""
        result = SkillResult(
            status=SkillStatus.PARTIAL,
            message="Partial message"
        )
        assert result.success is False  # Only SUCCESS is considered success

    def test_get_message_fallback(self):
        """Test message fallback when Korean not available."""
        result = SkillResult(
            status=SkillStatus.SUCCESS,
            message="English only"
        )
        # Should return English when Korean not available
        assert result.get_message("ko") == "English only"


class TestSkill:
    """Tests for Skill class."""

    def test_skill_creation(self):
        """Test creating a skill."""
        def handler(**params):
            return SkillResult(status=SkillStatus.SUCCESS)

        skill = Skill(
            name="test_skill",
            description="Test skill description",
            parameters={"param1": {"type": "string"}},
            required_params=["param1"],
            handler=handler
        )
        assert skill.name == "test_skill"
        assert skill.description == "Test skill description"
        assert "param1" in skill.parameters

    def test_skill_to_tool_definition(self):
        """Test converting skill to tool definition."""
        def handler(**params):
            return SkillResult(
                status=SkillStatus.SUCCESS,
                data={"received": params.get("value")}
            )

        skill = Skill(
            name="test_skill",
            description="Test",
            parameters={"value": {"type": "integer"}},
            handler=handler
        )
        tool_def = skill.to_tool_definition()
        assert tool_def.name == "test_skill"
        assert tool_def.description == "Test"


class TestToolDefinition:
    """Tests for ToolDefinition class."""

    def test_tool_definition(self):
        """Test creating a tool definition."""
        tool = ToolDefinition(
            name="test_tool",
            description="Test tool",
            parameters={"param1": {"type": "string"}}
        )
        assert tool.name == "test_tool"
        assert tool.description == "Test tool"


class TestAgentResponse:
    """Tests for AgentResponse class."""

    def test_agent_response_success(self):
        """Test creating a successful response."""
        response = AgentResponse(
            success=True,
            message="Success",
            message_ko="성공",
            agent_name="TestAgent"
        )
        assert response.success
        assert response.get_message("en") == "Success"
        assert response.get_message("ko") == "성공"

    def test_agent_response_with_results(self):
        """Test response with skill results."""
        result1 = SkillResult(status=SkillStatus.SUCCESS, message="Step 1")
        result2 = SkillResult(status=SkillStatus.SUCCESS, message="Step 2")

        response = AgentResponse(
            success=True,
            message="Workflow complete",
            results=[result1, result2]
        )
        assert len(response.results) == 2


class TestWorkflowStep:
    """Tests for WorkflowStep class."""

    def test_workflow_step(self):
        """Test creating a workflow step."""
        step = WorkflowStep(
            agent_name="flux_analysis",
            skill_name="perform_fba",
            params={"fraction": 0.9}
        )
        assert step.agent_name == "flux_analysis"
        assert step.skill_name == "perform_fba"
        assert step.params["fraction"] == 0.9


class TestAgentContext:
    """Tests for AgentContext class."""

    def test_context_creation(self, mock_appdata):
        """Test creating an agent context."""
        context = AgentContext(
            appdata=mock_appdata,
            main_window=None,
        )
        assert context.appdata is mock_appdata
        assert context.current_language == "en"

    def test_conversation_history(self, mock_appdata):
        """Test conversation history management."""
        context = AgentContext(appdata=mock_appdata)
        context.add_message("user", "Hello")
        context.add_message("assistant", "Hi there")

        assert len(context.conversation_history) == 2
        assert context.conversation_history[0]["role"] == "user"
        assert context.conversation_history[1]["role"] == "assistant"

    def test_analysis_results_cache(self, mock_appdata):
        """Test analysis results caching."""
        context = AgentContext(appdata=mock_appdata)
        context.cache_result("fba", {"growth_rate": 0.87})

        result = context.get_cached_result("fba")
        assert result is not None
        assert result["growth_rate"] == 0.87

    def test_clear_cache(self, mock_appdata):
        """Test clearing analysis cache."""
        context = AgentContext(appdata=mock_appdata)
        context.cache_result("fba", {"growth_rate": 0.87})
        context.analysis_results.clear()  # Direct clear

        assert context.get_cached_result("fba") is None
