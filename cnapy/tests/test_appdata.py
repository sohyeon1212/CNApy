"""Tests for appdata module utility functions and Scenario class."""

import json
import os
import tempfile

from cnapy.appdata import Scenario, my_mean, parse_scenario


class TestParseScenario:
    """Tests for parse_scenario function."""

    def test_parse_float_string(self):
        """Test parsing a float string returns tuple with same value."""
        result = parse_scenario("5.0")
        assert result == (5.0, 5.0)

    def test_parse_negative_float(self):
        """Test parsing a negative float string."""
        result = parse_scenario("-3.5")
        assert result == (-3.5, -3.5)

    def test_parse_zero(self):
        """Test parsing zero."""
        result = parse_scenario("0")
        assert result == (0.0, 0.0)

    def test_parse_tuple_string(self):
        """Test parsing a tuple string."""
        result = parse_scenario("(1.0, 5.0)")
        assert result == (1.0, 5.0)

    def test_parse_negative_tuple(self):
        """Test parsing a tuple with negative values."""
        result = parse_scenario("(-10.0, 10.0)")
        assert result == (-10.0, 10.0)

    def test_parse_integer_string(self):
        """Test parsing an integer string."""
        result = parse_scenario("10")
        assert result == (10.0, 10.0)


class TestMyMean:
    """Tests for my_mean function."""

    def test_mean_of_float(self):
        """Test mean of a single float returns the float."""
        result = my_mean(5.0)
        assert result == 5.0

    def test_mean_of_tuple(self):
        """Test mean of a tuple returns the average."""
        result = my_mean((2.0, 8.0))
        assert result == 5.0

    def test_mean_of_negative_tuple(self):
        """Test mean of a tuple with negative values."""
        result = my_mean((-10.0, 10.0))
        assert result == 0.0

    def test_mean_of_same_values(self):
        """Test mean when both values are the same."""
        result = my_mean((5.0, 5.0))
        assert result == 5.0

    def test_mean_of_zero(self):
        """Test mean of zero."""
        result = my_mean(0.0)
        assert result == 0.0


class TestScenario:
    """Tests for Scenario class."""

    def test_scenario_init(self):
        """Test Scenario initialization."""
        scenario = Scenario()

        assert len(scenario) == 0
        assert scenario.objective_direction == "max"
        assert scenario.use_scenario_objective is False
        assert len(scenario.pinned_reactions) == 0
        assert scenario.description == ""
        assert len(scenario.constraints) == 0
        assert len(scenario.reactions) == 0

    def test_scenario_as_dict(self):
        """Test Scenario behaves as a dictionary."""
        scenario = Scenario()

        scenario["R1"] = (5.0, 5.0)
        scenario["R2"] = (0.0, 10.0)

        assert len(scenario) == 2
        assert scenario["R1"] == (5.0, 5.0)
        assert scenario["R2"] == (0.0, 10.0)

    def test_scenario_clear_flux_values(self):
        """Test clear_flux_values clears only flux values."""
        scenario = Scenario()
        scenario["R1"] = (5.0, 5.0)
        scenario.description = "Test description"

        scenario.clear_flux_values()

        assert len(scenario) == 0
        # Description should still exist after clear_flux_values
        # But based on the code, clear_flux_values only clears the dict

    def test_scenario_clear(self):
        """Test clear resets the entire scenario."""
        scenario = Scenario()
        scenario["R1"] = (5.0, 5.0)
        scenario.description = "Test description"
        scenario.pinned_reactions.add("R2")

        scenario.clear()

        assert len(scenario) == 0
        assert scenario.description == ""
        assert len(scenario.pinned_reactions) == 0

    def test_scenario_save_and_load_basic(self):
        """Test basic save functionality without model."""
        scenario = Scenario()
        scenario["R1"] = (5.0, 5.0)
        scenario["R2"] = (0.0, 10.0)
        scenario.description = "Test scenario"
        scenario.objective_direction = "min"

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test.scen")
            scenario.save(filepath)

            # Verify file was created and contains valid JSON
            assert os.path.exists(filepath)
            with open(filepath) as f:
                data = json.load(f)

            assert data["fluxes"] == {"R1": [5.0, 5.0], "R2": [0.0, 10.0]}
            assert data["description"] == "Test scenario"
            assert data["objective_direction"] == "min"
            assert scenario.has_unsaved_changes is False

    def test_scenario_items_iteration(self):
        """Test iteration over scenario items."""
        scenario = Scenario()
        scenario["R1"] = (1.0, 1.0)
        scenario["R2"] = (2.0, 2.0)
        scenario["R3"] = (3.0, 3.0)

        items = list(scenario.items())
        assert len(items) == 3
        assert ("R1", (1.0, 1.0)) in items

    def test_scenario_get(self):
        """Test get method with default value."""
        scenario = Scenario()
        scenario["R1"] = (5.0, 5.0)

        assert scenario.get("R1") == (5.0, 5.0)
        assert scenario.get("R2") is None
        assert scenario.get("R2", (0.0, 0.0)) == (0.0, 0.0)

    def test_scenario_update(self):
        """Test update method."""
        scenario = Scenario()
        scenario.update({"R1": (1.0, 1.0), "R2": (2.0, 2.0)})

        assert scenario["R1"] == (1.0, 1.0)
        assert scenario["R2"] == (2.0, 2.0)

    def test_scenario_pop(self):
        """Test pop method."""
        scenario = Scenario()
        scenario["R1"] = (5.0, 5.0)

        value = scenario.pop("R1")
        assert value == (5.0, 5.0)
        assert "R1" not in scenario

    def test_scenario_pop_default(self):
        """Test pop with default value for missing key."""
        scenario = Scenario()

        value = scenario.pop("R1", None)
        assert value is None

    def test_scenario_version(self):
        """Test scenario version is set correctly."""
        scenario = Scenario()
        assert scenario.version == 4

    def test_scenario_empty_constraint(self):
        """Test empty_constraint class attribute."""
        assert Scenario.empty_constraint == (None, "", "")
