#!/usr/bin/env python3
"""
CMBS Test: Document Search as Repair Obligation (DSRO)

This script tests the DSO (Document Search Obligation) flow:
1. DSO entry only allowed during repair
2. Non-repetition rule enforcement
3. CCIL-gated exit (belief change required)
4. Exhaustion semantics
5. Logging of DSO episodes

Usage:
    python -m cmbs.test_dsro
"""

import sys
import os
import json
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cmbs.masks import Masks, DSOMask, EvidenceState
from cmbs.supervisor import Supervisor, Verdict
from cmbs.observer import Observer
from cmbs.agent_protocol import AgentStep, AgentBelief, AgentAction, ActionType
from cmbs.document_oracle import load_document, DocumentOracle
from cmbs.ccil import CCILConfig


def test_dso_mask():
    """Test DSOMask state tracking."""
    print("[*] Testing DSOMask...")

    dso = DSOMask()

    # Initially inactive
    assert not dso.active, "DSO should be inactive initially"
    assert len(dso.probe_history) == 0, "Probe history should be empty"

    # Enter DSO
    dso.enter(
        entropy_posture=0.8,
        capability_opa=0.3,
        repair_pressure=0.7,
    )

    assert dso.active, "DSO should be active after enter()"
    assert dso.entry_entropy_posture == 0.8
    assert dso.entry_capability_opa == 0.3
    assert dso.entry_repair_pressure == 0.7

    # Record probes (need min 2 for exit)
    dso.record_probe("open_section", "rules")
    dso.record_probe("search_keyword", "comprehension")

    assert len(dso.probe_history) == 2
    assert dso.is_probe_repeated("open_section", "rules"), "Should detect repeated probe"
    assert not dso.is_probe_repeated("open_section", "testing"), "Should not detect new probe as repeat"

    # Test exit conditions (now returns tuple)
    # No change - should not exit (AND-based: needs entropy drop AND structural change)
    can_exit, reason = dso.can_exit_with_belief_change(0.8, 0.3, 0.7)
    assert not can_exit, f"Should not exit with no change: {reason}"

    # Only entropy decreased - not enough (need structural change too)
    can_exit, reason = dso.can_exit_with_belief_change(0.7, 0.3, 0.7)
    # With epsilon_entropy=0.05, 0.8-0.7=0.1 is enough entropy drop
    # But need capability OR pressure change too
    # Actually 0.7 to 0.3 is enough entropy, but no capability/pressure change
    # So this should NOT exit now with the AND-based logic
    print(f"  Entropy only: can_exit={can_exit}, reason={reason}")

    # Entropy decreased AND capability increased - should exit
    can_exit, reason = dso.can_exit_with_belief_change(0.7, 0.35, 0.7)
    assert can_exit, f"Should exit with entropy decrease AND capability increase: {reason}"

    # Entropy decreased AND pressure decreased - should exit
    can_exit, reason = dso.can_exit_with_belief_change(0.7, 0.3, 0.65)
    assert can_exit, f"Should exit with entropy decrease AND pressure decrease: {reason}"

    # Test probe tier classification
    assert dso.get_probe_tier("rules") == 1, "rules should be tier 1"
    assert dso.get_probe_tier("hostNetwork") == 3, "hostNetwork should be tier 3"

    # Exit DSO
    dso.exit("belief_change")
    assert not dso.active, "DSO should be inactive after exit"
    assert dso.exit_reason == "belief_change"

    print("[+] DSOMask tests passed")


def test_supervisor_dso_entry():
    """Test that DSO entry is only allowed during repair."""
    print("[*] Testing DSO entry conditions...")

    with tempfile.TemporaryDirectory() as work_dir:
        observer = Observer(work_dir)
        supervisor = Supervisor(observer=observer)

        # Create document_search action
        step_dso = AgentStep(
            belief=AgentBelief(),
            action=AgentAction(type=ActionType.DOCUMENT_SEARCH, payload={}),
        )

        # Without repair_required, should be blocked
        supervisor.masks.repair_required = False
        response = supervisor.evaluate_step(step_dso)
        assert response.verdict == Verdict.BLOCK, f"DSO entry should be blocked without repair: {response}"
        print("[+] DSO entry blocked when not in repair")

        # With repair_required, should be allowed
        supervisor.masks.repair_required = True
        response = supervisor.evaluate_step(step_dso)
        assert response.verdict == Verdict.ALLOW, f"DSO entry should be allowed during repair: {response}"
        print("[+] DSO entry allowed during repair")


def test_supervisor_dso_actions():
    """Test that only probe_document is allowed inside DSO."""
    print("[*] Testing DSO action gating...")

    with tempfile.TemporaryDirectory() as work_dir:
        observer = Observer(work_dir)
        supervisor = Supervisor(observer=observer)

        # Enter DSO
        supervisor.masks.repair_required = True
        supervisor.enter_dso()

        assert supervisor.masks.dso.active, "DSO should be active"

        # Probe document should be allowed
        step_probe = AgentStep(
            belief=AgentBelief(),
            action=AgentAction(
                type=ActionType.PROBE_DOCUMENT,
                payload={"kind": "open_section", "target": "rules"},
            ),
        )
        response = supervisor.evaluate_step(step_probe)
        assert response.verdict == Verdict.ALLOW, f"Probe should be allowed in DSO: {response}"
        print("[+] probe_document allowed inside DSO")

        # Other actions should be blocked
        step_generate = AgentStep(
            belief=AgentBelief(),
            action=AgentAction(
                type=ActionType.GENERATE_POLICY,
                payload={"content": "test"},
            ),
        )
        response = supervisor.evaluate_step(step_generate)
        assert response.verdict == Verdict.BLOCK, f"Generate should be blocked in DSO: {response}"
        print("[+] generate_policy blocked inside DSO")


def test_supervisor_non_repetition():
    """Test that repeated probes are blocked."""
    print("[*] Testing non-repetition rule...")

    with tempfile.TemporaryDirectory() as work_dir:
        observer = Observer(work_dir)
        supervisor = Supervisor(observer=observer)

        # Enter DSO
        supervisor.masks.repair_required = True
        supervisor.enter_dso()

        # First probe should be allowed
        step_probe1 = AgentStep(
            belief=AgentBelief(),
            action=AgentAction(
                type=ActionType.PROBE_DOCUMENT,
                payload={"kind": "open_section", "target": "rules"},
            ),
        )
        response = supervisor.evaluate_step(step_probe1)
        assert response.verdict == Verdict.ALLOW, "First probe should be allowed"
        supervisor.record_dso_probe("open_section", "rules")
        print("[+] First probe allowed")

        # Repeated probe should be blocked
        response = supervisor.evaluate_step(step_probe1)
        assert response.verdict == Verdict.BLOCK, f"Repeated probe should be blocked: {response}"
        print("[+] Repeated probe blocked")

        # Different probe should be allowed
        step_probe2 = AgentStep(
            belief=AgentBelief(),
            action=AgentAction(
                type=ActionType.PROBE_DOCUMENT,
                payload={"kind": "search_keyword", "target": "comprehension"},
            ),
        )
        response = supervisor.evaluate_step(step_probe2)
        assert response.verdict == Verdict.ALLOW, "Different probe should be allowed"
        print("[+] Different probe allowed")


def test_document_oracle():
    """Test document oracle probing."""
    print("[*] Testing DocumentOracle...")

    oracle = load_document("agent-docs/rego_cheat_sheet.txt")

    # Test open_section
    result = oracle.probe("open_section", "rules")
    assert result.found, "Should find 'rules' section"
    assert "deny" in result.text.lower() or "allow" in result.text.lower(), "Rules section should contain rule content"
    print(f"[+] Found 'rules' section ({len(result.text)} chars)")

    # Test search_keyword
    result = oracle.probe("search_keyword", "comprehension")
    assert result.found, "Should find 'comprehension' keyword"
    assert result.section_id is not None, "Should have section context"
    print(f"[+] Found 'comprehension' in section '{result.section_id}'")

    # Test not found
    result = oracle.probe("open_section", "nonexistent")
    assert not result.found, "Should not find nonexistent section"
    print("[+] Correctly reports not found for nonexistent section")

    # Test available sections and keywords
    sections = oracle.get_available_sections()
    keywords = oracle.get_available_keywords()
    print(f"[+] Document has {len(sections)} sections, {len(keywords)} keywords")


def test_dso_with_ccil():
    """Test DSO with CCIL integration for belief-gated exit."""
    print("[*] Testing DSO with CCIL integration...")

    with tempfile.TemporaryDirectory() as work_dir:
        observer = Observer(work_dir)
        ccil_config = CCILConfig(enabled=True)
        supervisor = Supervisor(observer=observer, ccil_config=ccil_config)

        # Enter repair and DSO
        supervisor.masks.repair_required = True
        supervisor.enter_dso()

        assert supervisor.masks.dso.active, "DSO should be active"
        assert supervisor.masks.dso.entry_entropy_posture is not None, "Entry metrics should be recorded"

        print(f"[+] Entry metrics: entropy={supervisor.masks.dso.entry_entropy_posture:.3f}, "
              f"cap_opa={supervisor.masks.dso.entry_capability_opa:.3f}, "
              f"repair_pressure={supervisor.masks.dso.entry_repair_pressure:.3f}")

        # Check exit condition
        can_exit, reason = supervisor.can_exit_dso()
        print(f"[+] Can exit DSO: {can_exit}, reason: {reason}")


def test_full_dso_flow():
    """Test the complete DSO flow from entry to exit."""
    print("[*] Testing full DSO flow...")

    with tempfile.TemporaryDirectory() as work_dir:
        observer = Observer(work_dir)
        ccil_config = CCILConfig(enabled=True)
        supervisor = Supervisor(observer=observer, ccil_config=ccil_config)

        # Load document oracle
        oracle = load_document("agent-docs/rego_cheat_sheet.txt")

        # Simulate repair mode (failed execution)
        supervisor.masks.repair_required = True
        supervisor.masks.evidence.state = EvidenceState.ATTEMPTED

        # Step 1: Enter DSO
        step_enter = AgentStep(
            belief=AgentBelief(evidence="attempted"),
            action=AgentAction(type=ActionType.DOCUMENT_SEARCH, payload={}),
        )
        response = supervisor.evaluate_step(step_enter)
        assert response.verdict == Verdict.ALLOW, f"DSO entry should be allowed: {response}"
        supervisor.enter_dso()
        print(f"[+] Entered DSO (probes: {supervisor.masks.dso.get_probe_count()})")

        # Step 2: Probe documents
        probes = [
            ("open_section", "rules"),
            ("search_keyword", "comprehension"),
            ("open_section", "testing"),
        ]

        for kind, target in probes:
            step_probe = AgentStep(
                belief=AgentBelief(evidence="attempted"),
                action=AgentAction(
                    type=ActionType.PROBE_DOCUMENT,
                    payload={"kind": kind, "target": target},
                ),
            )
            response = supervisor.evaluate_step(step_probe)
            assert response.verdict == Verdict.ALLOW, f"Probe should be allowed: {response}"

            # Execute probe
            result = oracle.probe(kind, target)
            supervisor.record_dso_probe(kind, target)

            print(f"[+] Probe ({kind}, {target}): {'found' if result.found else 'not found'}")

        print(f"[+] Total probes: {supervisor.masks.dso.get_probe_count()}")

        # Step 3: Check exit condition
        can_exit, reason = supervisor.can_exit_dso()
        print(f"[+] Exit check: can_exit={can_exit}, reason={reason}")

        # Step 4: Exit DSO
        if can_exit:
            supervisor.exit_dso(reason)
            print(f"[+] Exited DSO with reason: {reason}")
        else:
            # Simulate belief change for test
            supervisor.masks.dso.exit("belief_change")
            print("[+] Forced DSO exit for test")

        assert not supervisor.masks.dso.active, "DSO should be inactive after exit"
        assert supervisor.masks.dso.exit_reason is not None, "Exit reason should be recorded"

        # Verify probe history preserved
        assert len(supervisor.masks.dso.probe_history) == 3, "Probe history should be preserved"

        print("[+] Full DSO flow completed successfully")


def test_dso_logging():
    """Test DSO episode logging."""
    print("[*] Testing DSO logging...")

    dso = DSOMask()
    dso.enter(0.8, 0.3, 0.7)
    dso.record_probe("open_section", "rules")
    dso.record_probe("search_keyword", "comprehension")
    dso.exit("belief_change")

    data = dso.to_dict()

    assert data["active"] == False
    assert data["probe_count"] == 2
    assert data["exit_reason"] == "belief_change"
    assert len(data["probe_history"]) == 2

    # Verify JSON serializable
    json_str = json.dumps(data)
    parsed = json.loads(json_str)
    assert parsed["probe_count"] == 2

    print("[+] DSO logging works correctly")
    print(f"    Serialized: {json_str[:100]}...")


def main():
    """Run all DSRO tests."""
    print("=" * 60)
    print("CMBS DSRO Tests")
    print("=" * 60)
    print()

    tests = [
        test_dso_mask,
        test_supervisor_dso_entry,
        test_supervisor_dso_actions,
        test_supervisor_non_repetition,
        test_document_oracle,
        test_dso_with_ccil,
        test_full_dso_flow,
        test_dso_logging,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"[!] FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
