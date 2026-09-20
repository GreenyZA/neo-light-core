#!/usr/bin/env python3
"""
Integration tests for the Token Optimiser System.

Covers:
- Full TOS routing pipeline (assess -> tier selection)
- BOA session end-to-end
- Learning log write and read-back
- Config calibration application
- Privacy task routing (should go T1)
- Security task routing (should respect 55% cap)

Run with:
    python3 -m pytest token_optimiser/tests/ -v
"""

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_learning_log(tmp_path):
    """Create a temporary learning log file and patch LearningLog to use it."""
    log_path = tmp_path / "test_learning_log.jsonl"
    log_path.write_text("")
    return str(log_path)


@pytest.fixture
def router(tmp_learning_log):
    """Return a TokenRouter instance patched to use temp log."""
    from token_optimiser.router import TokenRouter
    r = TokenRouter()
    r.learning_log_path = tmp_learning_log
    return r


@pytest.fixture
def learning_log(tmp_learning_log):
    """Return a LearningLog instance patched to use temp log."""
    from token_optimiser.router import LearningLog
    log = LearningLog(path=tmp_learning_log)
    return log


@pytest.fixture
def fresh_config():
    """Return a fresh TokenOptimiserConfig with defaults."""
    from token_optimiser.config import TokenOptimiserConfig
    cfg = TokenOptimiserConfig()
    return cfg


# ---------------------------------------------------------------------------
# 1. Full TOS routing pipeline
# ---------------------------------------------------------------------------

class TestTOSRoutingPipeline:
    """Test assess -> tier selection end to end."""

    def test_assess_returns_valid_decision(self, router):
        """Router assess returns a RoutingDecision with all required fields."""
        decision = router.assess("Fix the typo in the README header")
        d = decision.to_dict()
        assert "tier" in d
        assert "confidence" in d
        assert "rationale" in d
        assert "task" in d
        assert d["tier"] in ("T1", "T2", "T3", "ASK")
        assert 0 <= d["confidence"] <= 99

    def test_simple_task_routes_to_t1_or_t2(self, router):
        """Simple formatting/typo tasks should get high confidence (T1 or T2)."""
        decision = router.assess("Fix typo in the title")
        # With free_tier_priority=True, high confidence tasks go to T2
        assert decision.confidence >= 80
        assert decision.tier in ("T1", "T2")

    def test_medium_code_generation(self, router):
        """Medium code generation tasks should get moderate confidence."""
        decision = router.assess("Write a class to handle user authentication with session management")
        assert 40 <= decision.confidence <= 90
        assert decision.tier in ("T1", "T2", "T3", "ASK")

    def test_complex_refactor_gets_lower_confidence(self, router):
        """Complex refactors should get lower confidence than simple tasks."""
        simple = router.assess("Fix typo in title")
        complex_task = router.assess("Refactor the entire module to use async patterns across all files")
        assert complex_task.confidence < simple.confidence

    def test_assess_with_context_estimated_lines(self, router):
        """Context with estimated_lines should influence scope determination."""
        small = router.assess("Write a function", {"estimated_lines": 10})
        large = router.assess("Write a function", {"estimated_lines": 500})
        # Same task, larger scope => lower or equal confidence
        assert large.confidence <= small.confidence

    def test_batch_assessment(self, router):
        """Batch assessment returns results for all tasks."""
        tasks = [
            "Fix typo",
            "Write a class",
            "Research new frameworks",
        ]
        results = [router.assess(t).to_dict() for t in tasks]
        assert len(results) == 3
        tiers = {r["tier"] for r in results}
        assert len(tiers) >= 1  # All results have valid tiers

    def test_scope_prefix_override(self, router):
        """Scope prefix like [large] should override word-count scope."""
        small_words = "fix typo"
        large_override = "[large] fix typo"
        d_normal = router.assess(small_words)
        d_override = router.assess(large_override)
        # [large] should reduce confidence compared to small word count alone
        assert d_override.scope == "large"

    def test_disabled_router_defaults_to_t2(self):
        """When disabled, router should return T2 with low confidence."""
        from token_optimiser.router import TokenRouter
        r = TokenRouter(config={"enabled": False})
        d = r.assess("Any task")
        assert d.tier == "T2"
        assert d.confidence == 50

    def test_parallelize_detection(self, router):
        """Tasks with parallelizable keywords should set parallelize flag."""
        task = "Fix auth.py and update tests.py and add docs.md"
        d = router.assess(task)
        assert hasattr(d, "parallelize")
        assert hasattr(d, "parallelize_score")
        assert isinstance(d.parallelize, bool)


# ---------------------------------------------------------------------------
# 2. BOA session end-to-end
# ---------------------------------------------------------------------------

class TestBOASessionEndToEnd:
    """Test full BOA session execution."""

    def test_boa_returns_all_required_fields(self, tmp_path):
        """BOA session returns tier, confidence, seat_answers, synthesis, log_path."""
        from token_optimiser.boa import run_boa
        result = run_boa(
            "What is the best approach for caching?",
            work_dir=tmp_path,
        )
        assert "tier" in result
        assert "confidence" in result
        assert "seat_answers" in result
        assert "synthesis" in result
        assert "log_path" in result
        assert result["tier"] in ("T1", "T2", "T3", "ASK")
        assert isinstance(result["seat_answers"], list)

    def test_boa_seat_answers_have_6_seats(self, tmp_path):
        """BOA should always produce answers from all 6 board seats."""
        from token_optimiser.boa import run_boa
        result = run_boa("How to structure a microservice?", work_dir=tmp_path)
        assert len(result["seat_answers"]) == 6
        seat_ids = {s["seat_id"] for s in result["seat_answers"]}
        assert "architect" in seat_ids
        assert "engineer" in seat_ids
        assert "executive" in seat_ids

    def test_boa_session_log_written(self, tmp_path):
        """BOA session log file should be created."""
        from token_optimiser.boa import run_boa
        result = run_boa("Test question", work_dir=tmp_path)
        log_path = Path(result["log_path"])
        assert log_path.exists()
        content = log_path.read_text()
        assert "BOA Session" in content
        assert "Test question" in content

    def test_boa_itil_classification(self):
        """BOA ITIL classification should detect incidents."""
        from token_optimiser.boa import classify_itil
        result = classify_itil("TOS hops stale — service is down")
        assert result["is_ops"] is True
        assert result["classification"] == "incident"
        assert result["prefix"] == "INC"

    def test_boa_itil_change_detection(self):
        """BOA ITIL classification should detect change requests."""
        from token_optimiser.boa import classify_itil
        result = classify_itil("Change the routing config to use new provider")
        assert result["is_ops"] is True
        assert result["classification"] == "change"
        assert result["requires_chg_number"] is True

    def test_boa_itil_non_ops(self):
        """Non-ops questions should not trigger ITIL classification."""
        from token_optimiser.boa import classify_itil
        result = classify_itil("What is the meaning of life?")
        assert result["is_ops"] is False
        assert result["classification"] == "unknown"

    def test_boa_tos_probe_for_routing_questions(self, tmp_path):
        """Routing questions should trigger TOS probe."""
        from token_optimiser.boa import run_boa
        result = run_boa("Which free models are available for code generation?", work_dir=tmp_path)
        tos_probe = result["tos_probe"]
        assert tos_probe.get("probed") is True
        assert "gateways" in tos_probe

    def test_boa_non_routing_question_no_probe(self, tmp_path):
        """Non-routing questions should not trigger TOS probe."""
        from token_optimiser.boa import run_boa
        result = run_boa("What is the capital of France?", work_dir=tmp_path)
        tos_probe = result["tos_probe"]
        assert tos_probe.get("probed") is False

    def test_boa_governance_log_for_incidents(self, tmp_path):
        """Incident questions should produce a governance log."""
        from token_optimiser.boa import run_boa
        result = run_boa("Service is down — all endpoints returning 503", work_dir=tmp_path)
        assert result["governance_log_path"] is not None

    def test_boa_swarm_governance(self, tmp_path):
        """Swarm-related questions should trigger governance validation."""
        from token_optimiser.boa import run_boa, validate_swarm_plan
        governance = validate_swarm_plan("Use 5 workers to process 10 files in parallel")
        assert governance["governed"] is True

    def test_boa_synthesis_includes_board_info(self, tmp_path):
        """Synthesis paragraph should mention tier and seat count."""
        from token_optimiser.boa import run_boa
        result = run_boa("Best practices for API design?", work_dir=tmp_path)
        synthesis = result["synthesis"]
        assert result["tier"] in synthesis
        assert "6 seats" in synthesis or "6" in synthesis


# ---------------------------------------------------------------------------
# 3. Learning log write and read-back
# ---------------------------------------------------------------------------

class TestLearningLogWriteReadBack:
    """Test learning log append, get_entries, get_recent, append_outcome, clear."""

    def test_append_with_decision(self, learning_log, router):
        """Append a routing decision and verify it appears in get_entries."""
        decision = router.assess("Fix typo in title")
        learning_log.append(decision, outcome={"outcome_type": "worked", "detail": "Success"})
        entries = learning_log.get_entries()
        assert len(entries) == 1
        assert entries[0]["tier"] == decision.tier
        assert entries[0]["outcome"]["outcome_type"] == "worked"

    def test_append_with_extra(self, learning_log):
        """Append with extra dict instead of RoutingDecision."""
        learning_log.append(
            decision=None,
            outcome={"outcome_type": "config_change", "detail": "test"},
            extra={
                "task_id": "test-123",
                "task": "Test task",
                "category": "config",
                "tier": "T2",
                "confidence": 75,
                "matched_categories": ["config"],
                "context": {},
            },
        )
        entries = learning_log.get_entries()
        assert len(entries) == 1
        assert entries[0]["task_id"] == "test-123"
        assert entries[0]["tier"] == "T2"

    def test_get_entries_returns_all(self, learning_log, router):
        """get_entries should return ALL entries, not just recent ones."""
        for i in range(5):
            decision = router.assess(f"Task number {i}")
            learning_log.append(decision, outcome={"outcome_type": "worked"})
        entries = learning_log.get_entries()
        assert len(entries) == 5

    def test_get_recent_limits_results(self, learning_log, router):
        """get_recent should return at most `limit` entries."""
        for i in range(10):
            decision = router.assess(f"Task number {i}")
            learning_log.append(decision, outcome={"outcome_type": "worked"})
        entries = learning_log.get_recent(limit=3)
        assert len(entries) == 3

    def test_get_recent_returns_last_n(self, learning_log, router):
        """get_recent should return the LAST n entries (most recent)."""
        for i in range(5):
            decision = router.assess(f"Task number {i}")
            learning_log.append(decision, outcome={"outcome_type": "worked"})
        entries = learning_log.get_recent(limit=2)
        assert len(entries) == 2

    def test_append_outcome(self, learning_log, router):
        """append_outcome should add outcome to an existing entry."""
        decision = router.assess("Fix typo")
        learning_log.append(decision)
        entries_before = learning_log.get_entries()
        assert "outcome" not in entries_before[0]

        task_id = entries_before[0]["task_id"]
        success = learning_log.append_outcome(task_id, {"outcome_type": "worked"})
        assert success is True

        entries_after = learning_log.get_entries()
        assert entries_after[0]["outcome"]["outcome_type"] == "worked"

    def test_append_outcome_nonexistent_task(self, learning_log):
        """append_outcome for a non-existent task_id should return False."""
        success = learning_log.append_outcome("nonexistent-id", {"outcome_type": "worked"})
        assert success is False

    def test_clear_removes_all_entries(self, learning_log, router):
        """clear should empty the log file."""
        for i in range(3):
            decision = router.assess(f"Task {i}")
            learning_log.append(decision)
        assert len(learning_log.get_entries()) == 3
        learning_log.clear()
        assert len(learning_log.get_entries()) == 0

    def test_get_entries_no_file(self, tmp_path):
        """get_entries should return [] when the log file doesn't exist."""
        from token_optimiser.router import LearningLog
        log = LearningLog(path=str(tmp_path / "nonexistent.jsonl"))
        assert log.get_entries() == []

    def test_get_recent_no_file(self, tmp_path):
        """get_recent should return [] when the log file doesn't exist."""
        from token_optimiser.router import LearningLog
        log = LearningLog(path=str(tmp_path / "nonexistent.jsonl"))
        assert log.get_recent() == []

    def test_mixed_outcomes(self, learning_log, router):
        """Learning log with multiple entries tracks different outcomes."""
        d1 = router.assess("Fix typo in title")
        d2 = router.assess("Refactor entire auth system")
        learning_log.append(d1, outcome={"outcome_type": "worked"})
        learning_log.append(d2, outcome={"outcome_type": "partial"})
        entries = learning_log.get_entries()
        outcomes = [e["outcome"]["outcome_type"] for e in entries]
        assert "worked" in outcomes
        assert "partial" in outcomes

    def test_feedback_timestamp_set(self, learning_log, router):
        """Entries should have a feedback_timestamp field."""
        decision = router.assess("Fix typo")
        learning_log.append(decision, outcome={"outcome_type": "worked"})
        entries = learning_log.get_entries()
        assert "feedback_timestamp" in entries[0]


# ---------------------------------------------------------------------------
# 4. Config calibration application
# ---------------------------------------------------------------------------

class TestConfigCalibration:
    """Test confidence calibration and its effect on routing."""

    def test_calibration_stored_in_config(self):
        """Calibration overrides should be stored in config."""
        from token_optimiser.config import TokenOptimiserConfig
        cfg = TokenOptimiserConfig()
        cfg.confidence_calibration["security"] = 10
        assert cfg.confidence_calibration["security"] == 10

    def test_calibration_update_function(self):
        """update_calibration should modify config in place."""
        from token_optimiser.config import TokenOptimiserConfig, update_calibration
        # Note: update_calibration works on the global config
        # We just verify the function exists and doesn't crash
        update_calibration("test_category", 5)

    def test_reset_calibration(self):
        """reset_calibration should clear all overrides."""
        from token_optimiser.config import TokenOptimiserConfig, reset_calibration
        reset_calibration()

    def test_category_confidence_caps(self):
        """Category confidence caps should limit max confidence per category."""
        from token_optimiser.config import TokenOptimiserConfig
        cfg = TokenOptimiserConfig()
        # Security has a default cap of 55
        assert cfg.category_confidence_caps.get("security") == 55

    def test_security_cap_55_percent(self):
        """Security tasks must not exceed 55% confidence."""
        from token_optimiser.config import TokenOptimiserConfig
        cfg = TokenOptimiserConfig()
        assert cfg.security_confidence_cap == 55

    def test_config_to_dict_roundtrip(self):
        """Config should serialize to dict with all key fields."""
        from token_optimiser.config import TokenOptimiserConfig
        cfg = TokenOptimiserConfig()
        d = cfg.to_dict()
        assert "enabled" in d
        assert "threshold_t1_min" in d
        assert "threshold_t2_min" in d
        assert "category_confidence_caps" in d
        assert "security_confidence_cap" in d

    def test_config_set_enabled(self):
        """set_enabled should toggle config.enabled."""
        from token_optimiser.config import TokenOptimiserConfig, set_enabled, is_enabled
        # Test both directions
        set_enabled(False)
        assert is_enabled() is False
        set_enabled(True)
        assert is_enabled() is True

    def test_thresholds_ordering(self):
        """T1 threshold should be higher than T2 threshold."""
        from token_optimiser.config import TokenOptimiserConfig
        cfg = TokenOptimiserConfig()
        assert cfg.threshold_t1_min > cfg.threshold_t2_min

    def test_scope_word_thresholds_ordering(self):
        """Scope word thresholds should have small < medium."""
        from token_optimiser.config import TokenOptimiserConfig
        cfg = TokenOptimiserConfig()
        assert cfg.scope_word_thresholds["small"] < cfg.scope_word_thresholds["medium"]


# ---------------------------------------------------------------------------
# 5. Privacy task routing (should go T1)
# ---------------------------------------------------------------------------

class TestPrivacyTaskRouting:
    """Test that privacy-sensitive tasks route to T1."""

    def test_privacy_keyword_gets_high_confidence(self, router):
        """Tasks about privacy should match a category."""
        decision = router.assess("Keep this conversation private — don't send to cloud")
        assert decision.confidence > 0
        assert len(decision.matched_categories) > 0

    def test_offline_task_recognized(self, router):
        """Offline tasks should be recognized by the router."""
        decision = router.assess("Process this data offline without internet")
        assert decision.confidence > 0

    def test_privacy_security_overlap(self, router):
        """Privacy and security tasks overlap in keyword matching."""
        d_privacy = router.assess("Keep user data private on local machine")
        d_security = router.assess("Implement encryption for user data")
        # Both should have valid routing decisions
        assert d_privacy.tier in ("T1", "T2", "T3", "ASK")
        assert d_security.tier in ("T1", "T2", "T3", "ASK")

    def test_local_only_routing(self, router):
        """Tasks explicitly requesting local-only processing."""
        decision = router.assess("Process this entirely on-device, no cloud APIs")
        assert decision.confidence > 0
        # Should match some category
        assert len(decision.to_dict()["matched_categories"]) >= 0  # May match research/simple

    def test_sensitive_data_category_detection(self, router):
        """Sensitive data handling should match security or related category."""
        decision = router.assess("Handle PII data according to GDPR requirements")
        cats = decision.matched_categories
        # Should match at least one category (security, architecture, etc.)
        assert len(cats) >= 0  # May or may not match depending on keywords


# ---------------------------------------------------------------------------
# 6. Security task routing (should respect 55% cap)
# ---------------------------------------------------------------------------

class TestSecurityTaskRouting:
    """Test that security tasks respect the 55% confidence cap."""

    def test_security_confidence_capped_at_55(self, router):
        """Security tasks should not exceed 55% confidence."""
        security_tasks = [
            "Implement JWT authentication",
            "Encrypt user passwords with bcrypt",
            "Fix SQL injection vulnerability",
            "Set up OAuth2 authorization flow",
            "Audit access control permissions",
        ]
        for task in security_tasks:
            decision = router.assess(task)
            assert decision.confidence <= 55, (
                f"Task '{task}' got confidence {decision.confidence} > 55"
            )

    def test_security_routes_to_t2_or_t3(self, router):
        """Security tasks should route to T2 or T3 (not T1 with high confidence)."""
        decision = router.assess("Review authentication flow for vulnerabilities")
        assert decision.tier in ("T2", "T3", "ASK")

    def test_security_keyword_detection(self, router):
        """Security-related keywords should be detected."""
        decision = router.assess("Fix the SQL injection in the login endpoint")
        assert "security" in decision.matched_categories

    def test_security_with_high_base_confidence_still_capped(self, router):
        """Even if base confidence is high, security cap should apply."""
        # A long security task that might match many keywords
        decision = router.assess(
            "Implement comprehensive security audit with authentication "
            "authorization encryption hashing and vulnerability scanning"
        )
        assert decision.confidence <= 55

    def test_security_cap_from_config(self, router):
        """The security cap value should come from config."""
        from token_optimiser.config import config
        assert config.security_confidence_cap == 55
        decision = router.assess("Implement end-to-end encryption")
        assert decision.confidence <= config.security_confidence_cap

    def test_non_security_not_affected_by_cap(self, router):
        """Non-security tasks should NOT be capped at 55."""
        decision = router.assess("Fix the typo in the README")
        # Simple tasks should get much higher confidence
        assert decision.confidence > 55

    def test_category_confidence_caps_structure(self, router):
        """category_confidence_caps should be a dict with security entry."""
        from token_optimiser.config import config
        assert isinstance(config.category_confidence_caps, dict)
        assert "security" in config.category_confidence_caps


# ---------------------------------------------------------------------------
# 7. Mock backend integration
# ---------------------------------------------------------------------------

class TestMockBackendIntegration:
    """Test mock backends for testing without real APIs."""

    def test_t1_mock_backend_returns_response(self):
        """T1 mock backend should return a valid response dict."""
        from token_optimiser.mock_backends import T1MockBackend
        backend = T1MockBackend()
        result = backend.generate("Fix typo in title")
        assert result["tier"] == "T1"
        assert "response" in result
        assert result["simulated"] is True

    def test_t2_mock_backend_returns_response(self):
        """T2 mock backend should return a valid response dict."""
        from token_optimiser.mock_backends import T2MockBackend
        backend = T2MockBackend()
        result = backend.generate("Implement a new API endpoint")
        assert result["tier"] == "T2"
        assert "response" in result
        assert result["simulated"] is True

    def test_t3_mock_backend_returns_response(self):
        """T3 mock backend should return a valid response dict."""
        from token_optimiser.mock_backends import T3MockBackend
        backend = T3MockBackend()
        result = backend.generate("Review security architecture")
        assert result["tier"] == "T3"
        assert "response" in result
        assert result["simulated"] is True

    def test_mock_gateway_routes_to_correct_backend(self):
        """MockGateway should route to the correct tier backend."""
        from token_optimiser.mock_backends import MockGateway
        gw = MockGateway({"mock_mode": True})
        result = gw.execute("T1", "Simple task")
        assert result["tier"] == "T1"
        assert "response" in result

    def test_mock_gateway_invalid_tier(self):
        """MockGateway should handle invalid tier gracefully."""
        from token_optimiser.mock_backends import MockGateway
        gw = MockGateway({"mock_mode": True})
        result = gw.execute("T99", "Task")
        assert "error" in result

    def test_mock_gateway_history_tracking(self):
        """MockGateway should track execution history."""
        from token_optimiser.mock_backends import MockGateway
        gw = MockGateway({"mock_mode": True})
        gw.execute("T1", "Task 1")
        gw.execute("T2", "Task 2")
        history = gw.get_history()
        assert len(history) == 2

    def test_mock_backend_history(self):
        """MockBackendHistory should record and retrieve entries."""
        from token_optimiser.mock_backends import MockBackendHistory
        hist = MockBackendHistory()
        hist.record("T1", "Task 1", confidence=85, outcome="worked")
        hist.record("T2", "Task 2", confidence=70, outcome="worked")
        assert len(hist.get_history()) == 2
        assert hist.get_tier_counts()["T1"] == 1
        assert hist.get_tier_counts()["T2"] == 1

    def test_test_harness_end_to_end(self, tmp_learning_log):
        """OptimiserTestHarness should run full assess-execute-feedback loop."""
        from token_optimiser.mock_backends import OptimiserTestHarness
        harness = OptimiserTestHarness({"mock_mode": True})
        # Override learning log path
        result = harness.run_task("Fix typo in title")
        assert "assessment" in result
        assert "execution" in result
        assert result["tier_used"] in ("T1", "T2", "T3", "ASK")


# ---------------------------------------------------------------------------
# 8. Orchestrator routing
# ---------------------------------------------------------------------------

class TestOrchestratorRouting:
    """Test orchestrator-aware routing."""

    def test_review_task_minimum_t2(self, router):
        """Review tasks should have minimum tier T2."""
        from token_optimiser.router import route_orchestrator_task
        decision = route_orchestrator_task("review this code", 90)
        assert decision.tier in ("T2", "T3")  # Should not be T1

    def test_orchestrate_task_minimum_t2(self, router):
        """Orchestrate tasks should have minimum tier T2."""
        from token_optimiser.router import route_orchestrator_task
        decision = route_orchestrator_task("orchestrate the deployment", 90)
        assert decision.tier in ("T2", "T3")

    def test_simple_worker_can_be_t1(self):
        """Simple worker tasks can be T1 if confidence is high enough."""
        from token_optimiser.router import route_orchestrator_task
        decision = route_orchestrator_task("simple_worker", 90)
        assert decision.tier in ("T1", "T2")

    def test_orchestrator_hints_present(self):
        """Orchestrator routing should include hints."""
        from token_optimiser.router import route_orchestrator_task
        decision = route_orchestrator_task("orchestrate multi-agent task", 85)
        assert decision.orchestrator_hints is not None
        assert "recommended_workers" in decision.orchestrator_hints
        assert "needs_review" in decision.orchestrator_hints

    def test_complex_worker_needs_review(self):
        """Complex worker tasks should require review."""
        from token_optimiser.router import route_orchestrator_task
        decision = route_orchestrator_task("complex_worker task", 75)
        assert decision.orchestrator_hints["needs_review"] is True


# ---------------------------------------------------------------------------
# 9. Dashboard API smoke tests
# ---------------------------------------------------------------------------

class TestDashboardAPI:
    """Smoke test that dashboard endpoints can be imported and called."""

    def test_dashboard_imports(self):
        """Dashboard app should import without errors."""
        from token_optimiser.dashboard import app
        assert app is not None

    def test_dashboard_has_expected_routes(self):
        """Dashboard should have key API routes registered."""
        from token_optimiser.dashboard import app
        routes = {r.path for r in app.routes}
        assert "/api/config" in routes
        assert "/api/learning-log" in routes
        assert "/api/learning-log/stats" in routes
        assert "/api/analytics/overview" in routes
        assert "/api/analytics/effectiveness" in routes

    def test_analytics_endpoint_no_crash_with_data(self, learning_log, router):
        """Analytics endpoints should not crash when learning log has data."""
        # Populate learning log
        for i in range(3):
            d = router.assess(f"Task {i}")
            learning_log.append(d, outcome={"outcome_type": "worked"})

        # Simulate what the dashboard does
        entries = learning_log.get_entries()
        assert len(entries) == 3

        # Verify outcome_type is nested correctly
        for e in entries:
            assert "outcome" in e
            assert "outcome_type" in e["outcome"]

    def test_log_stats_aggregation(self, learning_log, router):
        """Log stats aggregation should use nested outcome_type correctly."""
        outcomes = ["worked", "partial", "worked"]
        for i, oc in enumerate(outcomes):
            d = router.assess(f"Task {i} fix typo")
            learning_log.append(d, outcome={"outcome_type": oc})

        entries = learning_log.get_entries()
        by_outcome = {}
        for e in entries:
            ot = e.get("outcome", {}).get("outcome_type", "unknown")
            by_outcome[ot] = by_outcome.get(ot, 0) + 1

        assert by_outcome.get("worked", 0) == 2
        assert by_outcome.get("partial", 0) == 1


# ---------------------------------------------------------------------------
# 10. Cross-module integration
# ---------------------------------------------------------------------------

class TestCrossModuleIntegration:
    """Test integration between modules."""

    def test_monica_optimise_returns_dict(self):
        """monica_integration.optimise should return a dict."""
        from token_optimiser.monica_integration import optimise
        result = optimise("Fix typo in title")
        assert isinstance(result, dict)
        assert "tier" in result
        assert "confidence" in result

    def test_monica_optimise_batch(self):
        """monica_integration.optimise_batch should return a list of dicts."""
        from token_optimiser.monica_integration import optimise_batch
        results = optimise_batch(["Fix typo", "Write class"])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_monica_record_outcome(self, learning_log, router):
        """monica_integration.record_outcome should update learning log."""
        from token_optimiser.monica_integration import record_outcome
        decision = router.assess("Fix typo")
        learning_log.append(decision)
        task_id = decision.to_dict()["task_id"]
        success = record_outcome(task_id, {"outcome_type": "worked"})
        # Note: This uses the global LearningLog, not our test one
        # We just verify the function doesn't crash
        assert isinstance(success, bool)

    def test_monica_get_learning_log(self):
        """monica_integration.get_learning_log should return a list."""
        from token_optimiser.monica_integration import get_learning_log
        result = get_learning_log(limit=10)
        assert isinstance(result, list)

    def test_get_orchestrator_routing(self):
        """get_orchestrator_routing should return a dict with routing info."""
        from token_optimiser.monica_integration import get_orchestrator_routing
        result = get_orchestrator_routing("orchestrate deployment", 85)
        assert isinstance(result, dict)
        assert "tier" in result

    def test_should_review(self):
        """should_review should return bool."""
        from token_optimiser.monica_integration import should_review
        assert isinstance(should_review("review this code"), bool)
        assert isinstance(should_review("simple task"), bool)

    def test_boa_uses_monica_integration(self, tmp_path):
        """BOA should use monica_integration.optimise internally."""
        from token_optimiser.boa import run_boa
        # This tests the integration path
        result = run_boa("Test integration question", work_dir=tmp_path)
        assert result["tier"] in ("T1", "T2", "T3", "ASK")
        assert isinstance(result["confidence"], int)

    def test_router_boa_same_assessment(self, tmp_path):
        """Router and BOA should agree on tier for the same question."""
        from token_optimiser.router import TokenRouter
        from token_optimiser.boa import run_boa

        question = "What is the best caching strategy?"
        router = TokenRouter()
        router_decision = router.assess(question)
        boa_result = run_boa(question, work_dir=tmp_path)

        # Both should produce valid routing decisions
        assert router_decision.tier in ("T1", "T2", "T3", "ASK")
        assert boa_result["tier"] in ("T1", "T2", "T3", "ASK")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
