"""
tests/test_agents.py
Phase 9 — Unit Tests for Real Estate Support Triage Agent

Run with:
    pytest tests/test_agents.py -v

Run with coverage:
    pip install pytest-cov
    pytest tests/test_agents.py -v --cov=agents --cov-report=term-missing
"""

import json
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Helpers — build a mock LLM response the same way LangChain returns one
# ---------------------------------------------------------------------------

def _mock_llm_response(content: str) -> MagicMock:
    """Return a mock object whose .content attribute equals *content*."""
    mock = MagicMock()
    mock.content = content
    return mock


# ===========================================================================
# ── CLASSIFIER TESTS (agents/classifier.py) ────────────────────────────────
# ===========================================================================

class TestClassifier:
    """Tests for classify_message() in agents/classifier.py"""

    # ── clean_json helper ──────────────────────────────────────────────────

    def test_clean_json_strips_markdown_fences(self):
        """clean_json must strip ```json … ``` wrappers Gemini sometimes adds."""
        from agents.classifier import clean_json
        raw = '```json\n{"urgency": "HIGH", "intent": "maintenance"}\n```'
        result = clean_json(raw)
        assert result == '{"urgency": "HIGH", "intent": "maintenance"}'

    def test_clean_json_passes_plain_json_unchanged(self):
        from agents.classifier import clean_json
        raw = '{"urgency": "LOW", "intent": "inquiry"}'
        assert clean_json(raw) == raw

    def test_clean_json_strips_whitespace(self):
        from agents.classifier import clean_json
        raw = '  {"urgency": "MEDIUM", "intent": "complaint"}  '
        result = clean_json(raw)
        parsed = json.loads(result)
        assert parsed["urgency"] == "MEDIUM"

    # ── classify_message — urgency detection ──────────────────────────────

    @patch("agents.classifier.llm")
    def test_classify_high_urgency_flooding(self, mock_llm):
        """Flooding messages should always be classified as HIGH / maintenance."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "HIGH", "intent": "maintenance"}'
        )
        from agents.classifier import classify_message
        result = classify_message("Water is flooding my apartment RIGHT NOW!")
        assert result["urgency"] == "HIGH"
        assert result["intent"] == "maintenance"

    @patch("agents.classifier.llm")
    def test_classify_high_urgency_no_heat(self, mock_llm):
        """No heat / heater broken → HIGH urgency."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "HIGH", "intent": "maintenance"}'
        )
        from agents.classifier import classify_message
        result = classify_message("My heater stopped working, it's freezing!")
        assert result["urgency"] == "HIGH"

    @patch("agents.classifier.llm")
    def test_classify_medium_urgency_broken_appliance(self, mock_llm):
        """Broken appliance → MEDIUM urgency."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "MEDIUM", "intent": "maintenance"}'
        )
        from agents.classifier import classify_message
        result = classify_message("My dishwasher is not draining properly.")
        assert result["urgency"] == "MEDIUM"
        assert result["intent"] == "maintenance"

    @patch("agents.classifier.llm")
    def test_classify_low_urgency_general_inquiry(self, mock_llm):
        """General questions → LOW urgency / inquiry."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "LOW", "intent": "inquiry"}'
        )
        from agents.classifier import classify_message
        result = classify_message("What are your office hours on weekends?")
        assert result["urgency"] == "LOW"
        assert result["intent"] == "inquiry"

    # ── classify_message — intent detection ───────────────────────────────

    @patch("agents.classifier.llm")
    def test_classify_intent_payment(self, mock_llm):
        """Rent / payment messages must map to the 'payment' intent."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "MEDIUM", "intent": "payment"}'
        )
        from agents.classifier import classify_message
        result = classify_message("I was charged twice for this month's rent.")
        assert result["intent"] == "payment"

    @patch("agents.classifier.llm")
    def test_classify_intent_viewing(self, mock_llm):
        """Scheduling a viewing → 'viewing' intent."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "LOW", "intent": "viewing"}'
        )
        from agents.classifier import classify_message
        result = classify_message("I'd like to schedule a viewing for Unit 4A.")
        assert result["intent"] == "viewing"

    @patch("agents.classifier.llm")
    def test_classify_intent_complaint(self, mock_llm):
        """Noise / neighbour complaints → 'complaint' intent."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "MEDIUM", "intent": "complaint"}'
        )
        from agents.classifier import classify_message
        result = classify_message("My upstairs neighbour is extremely noisy every night.")
        assert result["intent"] == "complaint"

    # ── classify_message — return-type contract ────────────────────────────

    @patch("agents.classifier.llm")
    def test_classify_returns_dict(self, mock_llm):
        """classify_message must always return a dict (not a string)."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "LOW", "intent": "inquiry"}'
        )
        from agents.classifier import classify_message
        result = classify_message("Hello, any updates?")
        assert isinstance(result, dict)

    @patch("agents.classifier.llm")
    def test_classify_contains_required_keys(self, mock_llm):
        """Result must always contain 'urgency' and 'intent' keys."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "LOW", "intent": "inquiry"}'
        )
        from agents.classifier import classify_message
        result = classify_message("Just checking in.")
        assert "urgency" in result
        assert "intent" in result

    @patch("agents.classifier.llm")
    def test_classify_urgency_values_are_valid(self, mock_llm):
        """urgency must be one of HIGH / MEDIUM / LOW."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "MEDIUM", "intent": "inquiry"}'
        )
        from agents.classifier import classify_message
        result = classify_message("I have a billing question.")
        assert result["urgency"] in {"HIGH", "MEDIUM", "LOW"}

    @patch("agents.classifier.llm")
    def test_classify_intent_values_are_valid(self, mock_llm):
        """intent must be one of the five defined categories."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "LOW", "intent": "inquiry"}'
        )
        from agents.classifier import classify_message
        result = classify_message("Do you allow pets?")
        assert result["intent"] in {
            "maintenance", "inquiry", "complaint", "payment", "viewing"
        }

    # ── edge cases ─────────────────────────────────────────────────────────

    @patch("agents.classifier.llm")
    def test_classify_handles_markdown_wrapped_response(self, mock_llm):
        """Pipeline must survive Gemini returning ```json … ``` fences."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '```json\n{"urgency": "HIGH", "intent": "maintenance"}\n```'
        )
        from agents.classifier import classify_message
        result = classify_message("Gas is leaking from the kitchen!")
        assert result["urgency"] == "HIGH"

    @patch("agents.classifier.llm")
    def test_classify_empty_message(self, mock_llm):
        """Empty-string input must still return a valid classification dict."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "LOW", "intent": "inquiry"}'
        )
        from agents.classifier import classify_message
        result = classify_message("")
        assert "urgency" in result and "intent" in result

    @patch("agents.classifier.llm")
    def test_classify_lockout_is_high_urgency(self, mock_llm):
        """Lockout scenario must be classified HIGH per urgency rules."""
        mock_llm.invoke.return_value = _mock_llm_response(
            '{"urgency": "HIGH", "intent": "maintenance"}'
        )
        from agents.classifier import classify_message
        result = classify_message("I'm locked out of my apartment, please help!")
        assert result["urgency"] == "HIGH"


# ===========================================================================
# ── NER EXTRACTOR TESTS (agents/ner_extractor.py) ──────────────────────────
# ===========================================================================

class TestNERExtractor:
    """Tests for extract_entities() in agents/ner_extractor.py"""

    # ── clean_json helper (same helper, different module) ─────────────────

    def test_clean_json_in_ner_module(self):
        from agents.ner_extractor import clean_json
        raw = '```\n{"tenant_id": "T-001"}\n```'
        result = clean_json(raw)
        assert '"tenant_id"' in result

    # ── entity extraction — positive cases ────────────────────────────────

    @patch("agents.ner_extractor.llm")
    def test_extract_tenant_id(self, mock_llm):
        """Tenant IDs (e.g. T-4421) must be extracted correctly."""
        mock_llm.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": "T-4421",
            "unit_number": "3B",
            "lease_date": None,
            "property_address": None,
            "person_name": None,
            "dates": None,
            "issue_type": "heater"
        }))
        from agents.ner_extractor import extract_entities
        result = extract_entities("I'm tenant #T-4421 in Unit 3B. My heater stopped working!")
        assert result["tenant_id"] == "T-4421"

    @patch("agents.ner_extractor.llm")
    def test_extract_unit_number(self, mock_llm):
        mock_llm.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": "T-4421", "unit_number": "3B",
            "lease_date": None, "property_address": None,
            "person_name": None, "dates": None, "issue_type": "heater"
        }))
        from agents.ner_extractor import extract_entities
        result = extract_entities("Unit 3B has a broken heater.")
        assert result["unit_number"] == "3B"

    @patch("agents.ner_extractor.llm")
    def test_extract_property_address(self, mock_llm):
        mock_llm.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": None, "unit_number": None,
            "lease_date": None,
            "property_address": "45 Maple Street, Springfield",
            "person_name": None, "dates": None, "issue_type": None
        }))
        from agents.ner_extractor import extract_entities
        result = extract_entities("My address is 45 Maple Street, Springfield.")
        assert result["property_address"] == "45 Maple Street, Springfield"

    @patch("agents.ner_extractor.llm")
    def test_extract_lease_date(self, mock_llm):
        mock_llm.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": None, "unit_number": None,
            "lease_date": "2024-01-01",
            "property_address": None,
            "person_name": None, "dates": "2024-01-01", "issue_type": None
        }))
        from agents.ner_extractor import extract_entities
        result = extract_entities("My lease started on January 1st, 2024.")
        assert result["lease_date"] == "2024-01-01"

    @patch("agents.ner_extractor.llm")
    def test_extract_person_name(self, mock_llm):
        mock_llm.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": None, "unit_number": None,
            "lease_date": None, "property_address": None,
            "person_name": "John Smith",
            "dates": None, "issue_type": None
        }))
        from agents.ner_extractor import extract_entities
        result = extract_entities("This is John Smith from Unit 7.")
        assert result["person_name"] == "John Smith"

    # ── entity extraction — null/missing fields ───────────────────────────

    @patch("agents.ner_extractor.llm")
    def test_extract_null_fields_when_absent(self, mock_llm):
        """Fields not found in the message must be null, not absent."""
        mock_llm.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": None, "unit_number": None,
            "lease_date": None, "property_address": None,
            "person_name": None, "dates": None, "issue_type": None
        }))
        from agents.ner_extractor import extract_entities
        result = extract_entities("Water is flooding my apartment RIGHT NOW!")
        assert result["tenant_id"] is None
        assert result["unit_number"] is None

    # ── return-type contract ──────────────────────────────────────────────

    @patch("agents.ner_extractor.llm")
    def test_extract_returns_dict(self, mock_llm):
        mock_llm.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": None, "unit_number": None,
            "lease_date": None, "property_address": None,
            "person_name": None, "dates": None, "issue_type": None
        }))
        from agents.ner_extractor import extract_entities
        result = extract_entities("Hello.")
        assert isinstance(result, dict)

    @patch("agents.ner_extractor.llm")
    def test_extract_contains_all_required_keys(self, mock_llm):
        """Result must include all 7 defined entity keys."""
        mock_llm.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": None, "unit_number": None,
            "lease_date": None, "property_address": None,
            "person_name": None, "dates": None, "issue_type": None
        }))
        from agents.ner_extractor import extract_entities
        result = extract_entities("Just a generic message.")
        required_keys = {
            "tenant_id", "unit_number", "lease_date",
            "property_address", "person_name", "dates", "issue_type"
        }
        assert required_keys.issubset(result.keys())

    # ── edge cases ─────────────────────────────────────────────────────────

    @patch("agents.ner_extractor.llm")
    def test_extract_handles_markdown_wrapped_response(self, mock_llm):
        payload = json.dumps({
            "tenant_id": "T-999", "unit_number": "5A",
            "lease_date": None, "property_address": None,
            "person_name": None, "dates": None, "issue_type": "flood"
        })
        mock_llm.invoke.return_value = _mock_llm_response(
            f"```json\n{payload}\n```"
        )
        from agents.ner_extractor import extract_entities
        result = extract_entities("T-999, Unit 5A — flooding!")
        assert result["tenant_id"] == "T-999"

    @patch("agents.ner_extractor.llm")
    def test_extract_multiple_entities_simultaneously(self, mock_llm):
        """All entities present in a single message must all be extracted."""
        mock_llm.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": "T-1001",
            "unit_number": "2C",
            "lease_date": "2023-06-15",
            "property_address": "12 Oak Avenue",
            "person_name": "Alice Brown",
            "dates": "2023-06-15",
            "issue_type": "broken lock"
        }))
        from agents.ner_extractor import extract_entities
        result = extract_entities(
            "Alice Brown (T-1001, Unit 2C, lease from 15 June 2023, "
            "12 Oak Avenue) reports a broken lock."
        )
        assert result["tenant_id"] == "T-1001"
        assert result["unit_number"] == "2C"
        assert result["property_address"] == "12 Oak Avenue"

    @patch("agents.ner_extractor.llm")
    def test_extract_empty_message(self, mock_llm):
        """Empty-string input must still return a valid dict."""
        mock_llm.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": None, "unit_number": None,
            "lease_date": None, "property_address": None,
            "person_name": None, "dates": None, "issue_type": None
        }))
        from agents.ner_extractor import extract_entities
        result = extract_entities("")
        assert isinstance(result, dict)


# ===========================================================================
# ── RESPONDER TESTS (agents/responder.py) ──────────────────────────────────
# ===========================================================================

class TestResponder:
    """Tests for generate_response() in agents/responder.py"""

    # ── response time commitments ─────────────────────────────────────────

    @patch("agents.responder.llm")
    def test_high_urgency_mentions_two_hours(self, mock_llm):
        """HIGH urgency draft must contain a 2-hour commitment."""
        draft = (
            "Dear Resident, we are treating this as a high-priority matter. "
            "A technician will be on-site within 2 hours."
        )
        mock_llm.invoke.return_value = _mock_llm_response(draft)
        from agents.responder import generate_response
        result = generate_response(
            "My heater stopped working!",
            {"urgency": "HIGH", "intent": "maintenance"},
            {"tenant_id": "T-001", "unit_number": "3B"}
        )
        assert "2 hour" in result.lower() or "two hour" in result.lower()

    @patch("agents.responder.llm")
    def test_medium_urgency_mentions_24_hours(self, mock_llm):
        """MEDIUM urgency draft must contain a 24-hour commitment."""
        draft = (
            "Thank you for letting us know. We will address this within 24 hours."
        )
        mock_llm.invoke.return_value = _mock_llm_response(draft)
        from agents.responder import generate_response
        result = generate_response(
            "My dishwasher is leaking.",
            {"urgency": "MEDIUM", "intent": "maintenance"},
            {"tenant_id": None, "unit_number": None}
        )
        assert "24 hour" in result.lower() or "twenty-four hour" in result.lower()

    @patch("agents.responder.llm")
    def test_low_urgency_mentions_business_days(self, mock_llm):
        """LOW urgency draft must mention business days."""
        draft = (
            "Hi, thanks for reaching out! We will respond within 3 business days."
        )
        mock_llm.invoke.return_value = _mock_llm_response(draft)
        from agents.responder import generate_response
        result = generate_response(
            "What are your office hours?",
            {"urgency": "LOW", "intent": "inquiry"},
            {"tenant_id": None, "unit_number": None}
        )
        assert "business day" in result.lower()

    # ── entity personalisation ────────────────────────────────────────────

    @patch("agents.responder.llm")
    def test_response_includes_unit_number(self, mock_llm):
        """Draft should reference the unit number when available."""
        draft = "Dear Tenant, we have noted your report for Unit 3B."
        mock_llm.invoke.return_value = _mock_llm_response(draft)
        from agents.responder import generate_response
        result = generate_response(
            "Heater broken in Unit 3B.",
            {"urgency": "HIGH", "intent": "maintenance"},
            {"tenant_id": "T-4421", "unit_number": "3B"}
        )
        assert "3B" in result or "3b" in result.lower()

    @patch("agents.responder.llm")
    def test_flooding_response_includes_emergency_info(self, mock_llm):
        """Flooding/fire responses should advise contacting emergency services."""
        draft = (
            "We are dispatching help immediately. "
            "If there is immediate danger, please call 911."
        )
        mock_llm.invoke.return_value = _mock_llm_response(draft)
        from agents.responder import generate_response
        result = generate_response(
            "Water is flooding my apartment RIGHT NOW!",
            {"urgency": "HIGH", "intent": "maintenance"},
            {"tenant_id": None, "unit_number": None}
        )
        assert "911" in result or "emergency" in result.lower()

    # ── return-type contract ──────────────────────────────────────────────

    @patch("agents.responder.llm")
    def test_response_returns_string(self, mock_llm):
        """generate_response must return a plain string."""
        mock_llm.invoke.return_value = _mock_llm_response("Some reply.")
        from agents.responder import generate_response
        result = generate_response(
            "General question",
            {"urgency": "LOW", "intent": "inquiry"},
            {}
        )
        assert isinstance(result, str)

    @patch("agents.responder.llm")
    def test_response_is_non_empty(self, mock_llm):
        """Draft reply must never be an empty string."""
        mock_llm.invoke.return_value = _mock_llm_response(
            "Thank you for your message. We will be in touch."
        )
        from agents.responder import generate_response
        result = generate_response(
            "Hello?",
            {"urgency": "LOW", "intent": "inquiry"},
            {}
        )
        assert len(result.strip()) > 0

    # ── tone & professionalism ────────────────────────────────────────────

    @patch("agents.responder.llm")
    def test_response_is_professional_tone(self, mock_llm):
        """Draft must contain a professional salutation or closing."""
        draft = (
            "Dear Resident, thank you for reaching out. "
            "Sincerely, Property Management Team."
        )
        mock_llm.invoke.return_value = _mock_llm_response(draft)
        from agents.responder import generate_response
        result = generate_response(
            "I have a complaint about noise.",
            {"urgency": "MEDIUM", "intent": "complaint"},
            {}
        )
        has_greeting = any(
            word in result.lower()
            for word in ["dear", "hi", "hello", "greetings"]
        )
        has_closing = any(
            word in result.lower()
            for word in ["sincerely", "regards", "team", "management"]
        )
        assert has_greeting or has_closing

    # ── edge cases ─────────────────────────────────────────────────────────

    @patch("agents.responder.llm")
    def test_response_with_unknown_tenant(self, mock_llm):
        """Missing tenant info (None) must not crash the responder."""
        mock_llm.invoke.return_value = _mock_llm_response(
            "Dear Resident, could you please provide your unit number?"
        )
        from agents.responder import generate_response
        result = generate_response(
            "Water is flooding!",
            {"urgency": "HIGH", "intent": "maintenance"},
            {"tenant_id": None, "unit_number": None}
        )
        assert isinstance(result, str) and len(result) > 0

    @patch("agents.responder.llm")
    def test_response_empty_entities_dict(self, mock_llm):
        """An empty entities dict must not raise a KeyError."""
        mock_llm.invoke.return_value = _mock_llm_response("Thank you for contacting us.")
        from agents.responder import generate_response
        result = generate_response(
            "Question about parking.",
            {"urgency": "LOW", "intent": "inquiry"},
            {}
        )
        assert isinstance(result, str)

    @patch("agents.responder.llm")
    def test_response_for_payment_intent(self, mock_llm):
        """Payment-related messages should produce a valid draft."""
        draft = (
            "Dear Tenant, thank you for flagging the billing issue. "
            "Our accounts team will review within 24 hours."
        )
        mock_llm.invoke.return_value = _mock_llm_response(draft)
        from agents.responder import generate_response
        result = generate_response(
            "I was charged twice this month.",
            {"urgency": "MEDIUM", "intent": "payment"},
            {"tenant_id": "T-777", "unit_number": "1A"}
        )
        assert len(result) > 0

    @patch("agents.responder.llm")
    def test_response_for_viewing_intent(self, mock_llm):
        """Viewing requests should produce a valid reply."""
        draft = "Hi, we would be happy to arrange a viewing within 3 business days."
        mock_llm.invoke.return_value = _mock_llm_response(draft)
        from agents.responder import generate_response
        result = generate_response(
            "Can I schedule a viewing for Unit 5B?",
            {"urgency": "LOW", "intent": "viewing"},
            {"tenant_id": None, "unit_number": "5B"}
        )
        assert isinstance(result, str) and len(result) > 0


# ===========================================================================
# ── PIPELINE INTEGRATION SMOKE TEST ────────────────────────────────────────
# ===========================================================================

class TestPipelineIntegration:
    """
    Lightweight smoke tests that wire all three agents together via
    main.triage_pipeline(), using mocks so no real API calls are made.
    """

    @patch("agents.responder.llm")
    @patch("agents.ner_extractor.llm")
    @patch("agents.classifier.llm")
    def test_full_pipeline_high_urgency(
        self, mock_clf, mock_ner, mock_resp
    ):
        mock_clf.invoke.return_value = _mock_llm_response(
            '{"urgency": "HIGH", "intent": "maintenance"}'
        )
        mock_ner.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": "T-4421", "unit_number": "3B",
            "lease_date": None, "property_address": None,
            "person_name": None, "dates": None, "issue_type": "heater"
        }))
        mock_resp.invoke.return_value = _mock_llm_response(
            "Dear T-4421, a technician will be on-site within 2 hours."
        )
        from main import triage_pipeline
        result = triage_pipeline(
            "Hi, I'm tenant #T-4421 in Unit 3B. My heater stopped working!"
        )
        assert result["urgency"] == "HIGH"
        assert result["intent"] == "maintenance"
        assert result["entities"]["tenant_id"] == "T-4421"
        assert "2 hour" in result["draft_reply"].lower()

    @patch("agents.responder.llm")
    @patch("agents.ner_extractor.llm")
    @patch("agents.classifier.llm")
    def test_full_pipeline_low_urgency(
        self, mock_clf, mock_ner, mock_resp
    ):
        mock_clf.invoke.return_value = _mock_llm_response(
            '{"urgency": "LOW", "intent": "inquiry"}'
        )
        mock_ner.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": None, "unit_number": None,
            "lease_date": None, "property_address": None,
            "person_name": None, "dates": None, "issue_type": None
        }))
        mock_resp.invoke.return_value = _mock_llm_response(
            "Hi, our office is open Monday to Friday. "
            "We will respond within 3 business days."
        )
        from main import triage_pipeline
        result = triage_pipeline("What are your office hours on weekends?")
        assert result["urgency"] == "LOW"
        assert result["intent"] == "inquiry"
        assert "business day" in result["draft_reply"].lower()

    @patch("agents.responder.llm")
    @patch("agents.ner_extractor.llm")
    @patch("agents.classifier.llm")
    def test_pipeline_result_has_all_keys(
        self, mock_clf, mock_ner, mock_resp
    ):
        """triage_pipeline result dict must contain all five expected keys."""
        mock_clf.invoke.return_value = _mock_llm_response(
            '{"urgency": "MEDIUM", "intent": "complaint"}'
        )
        mock_ner.invoke.return_value = _mock_llm_response(json.dumps({
            "tenant_id": None, "unit_number": None,
            "lease_date": None, "property_address": None,
            "person_name": None, "dates": None, "issue_type": "noise"
        }))
        mock_resp.invoke.return_value = _mock_llm_response(
            "We will investigate within 24 hours."
        )
        from main import triage_pipeline
        result = triage_pipeline("My neighbour is very noisy.")
        required_keys = {
            "original_message", "urgency", "intent",
            "entities", "draft_reply"
        }
        assert required_keys.issubset(result.keys())