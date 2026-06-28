"""
Enterprise Healthcare Prior Authorization Multi-Agent Core
-----------------------------------------------------------
Dual-mode (Local Ollama / AWS Bedrock) multi-agent system that converts
unstructured clinical notes + payer guidelines into a compliant
Medical Necessity Justification Letter or Peer-to-Peer Review Request.

Architecture (LangGraph state machine):

    START
      |
      v
    [Agent 1: Adversarial Extractor]  --> Pydantic-validated JSON
      |
      v
    <conditional: criteria_fully_met?>
      |                          |
      | True                     | False
      v                          v
    [Agent 2a: Standard       [Agent 2b: Peer-to-Peer
     Justification Letter]     Review Request]
      |                          |
      +-----------+--------------+
                  |
                  v
              [Streamlit HITL Gate: edit -> approve -> download]
                  |
                  v
                 END

Author: <your-name>
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Literal, Optional, TypedDict

import boto3
import requests
import streamlit as st
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
    NoCredentialsError,
)
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("pa_agent")

# ---------------------------------------------------------------------------
# Configuration (env-overridable for cloud deployment)
# ---------------------------------------------------------------------------
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_EXTRACTOR_MODEL = os.getenv("OLLAMA_EXTRACTOR_MODEL", "llama3.2:3b")
OLLAMA_WRITER_MODEL = os.getenv("OLLAMA_WRITER_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "180"))

BEDROCK_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
)
BEDROCK_READ_TIMEOUT = int(os.getenv("BEDROCK_READ_TIMEOUT", "60"))

# ---------------------------------------------------------------------------
# Pydantic schema -- the Agent 1 contract
# ---------------------------------------------------------------------------
class ExtractedClinicalCase(BaseModel):
    """Strict schema enforcing the handoff between Agent 1 and Agent 2."""

    primary_diagnosis: str = Field(..., description="Primary diagnosis in plain English.")
    icd10_codes: list[str] = Field(default_factory=list)
    requested_service: str = Field(..., description="Service/procedure/medication requested.")
    cpt_or_hcpcs_codes: list[str] = Field(default_factory=list)
    key_symptoms: list[str] = Field(default_factory=list)
    failed_conservative_treatments: list[str] = Field(default_factory=list)
    relevant_clinical_findings: list[str] = Field(default_factory=list)
    payer_criteria_summary: list[str] = Field(default_factory=list)
    criteria_fully_met: bool = Field(
        ...,
        description="True only if EVERY payer criterion is supported by the notes.",
    )
    unmet_criteria: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Inference backend abstraction (Strategy pattern)
# ---------------------------------------------------------------------------
class InferenceBackend:
    """Strategy interface so agents don't care where inference runs."""

    def generate(self, system: str, user: str, *, json_mode: bool = False,
                 model_override: Optional[str] = None) -> str:
        raise NotImplementedError


class OllamaBackend(InferenceBackend):
    """
    Self-hosted Ollama backend. Supports two-model strategy via model_override:
    - Agent 1 (extraction) uses a small fast model (llama3.2:3b)
    - Agent 2 (clinical prose) uses a larger model (llama3.2:3b)
    """

    def __init__(
        self,
        url: str = OLLAMA_URL,
        extractor_model: str = OLLAMA_EXTRACTOR_MODEL,
        writer_model: str = OLLAMA_WRITER_MODEL,
    ):
        self.url = url
        self.extractor_model = extractor_model
        self.writer_model = writer_model

    def generate(self, system, user, *, json_mode=False, model_override=None):
        model = model_override or self.extractor_model
        payload = {
            "model": model,
            "prompt": f"{system}\n\n---\n\n{user}",
            "stream": False,
            "options": {"temperature": 0.1},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            r = requests.post(self.url, json=payload, timeout=OLLAMA_TIMEOUT)
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.url}. Start it with `ollama serve` "
                f"and pull models with `ollama pull {model}`."
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(f"Ollama request timed out after {OLLAMA_TIMEOUT}s.") from e
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP error: {e}") from e


class BedrockBackend(InferenceBackend):
    """AWS Bedrock backend (documented production path; not active in cost-free deployment)."""

    def __init__(self, region: str = BEDROCK_REGION, model_id: str = BEDROCK_MODEL_ID):
        self.region = region
        self.model_id = model_id
        try:
            self.client = boto3.client(
                "bedrock-runtime",
                region_name=region,
                config=boto3.session.Config(
                    read_timeout=BEDROCK_READ_TIMEOUT,
                    retries={"max_attempts": 3, "mode": "standard"},
                ),
            )
        except NoCredentialsError as e:
            raise RuntimeError(
                "AWS credentials not found. Use IAM role on EC2 (recommended) or set "
                "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
            ) from e

    def generate(self, system, user, *, json_mode=False, model_override=None):
        if json_mode:
            user += "\n\nReturn ONLY a valid JSON object. No prose, no markdown fences."

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": 0.1,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }

        try:
            resp = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            payload = json.loads(resp["body"].read())
            return payload["content"][0]["text"].strip()
        except EndpointConnectionError as e:
            raise RuntimeError(f"Cannot reach Bedrock endpoint in {self.region}.") from e
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Unknown")
            msg = e.response.get("Error", {}).get("Message", str(e))
            if code in ("UnrecognizedClientException", "InvalidSignatureException"):
                raise RuntimeError(f"AWS auth failed ({code}): check credentials.") from e
            if code == "AccessDeniedException":
                raise RuntimeError(
                    f"Access denied to model {self.model_id}. Request access in "
                    "Bedrock console -> Model access."
                ) from e
            if code == "ThrottlingException":
                raise RuntimeError("Bedrock throttled the request. Retry shortly.") from e
            raise RuntimeError(f"Bedrock error ({code}): {msg}") from e
        except BotoCoreError as e:
            raise RuntimeError(f"AWS SDK error: {e}") from e


def make_backend(name: str) -> InferenceBackend:
    """Factory used by LangGraph nodes (which only see the state dict)."""
    if name == "ollama":
        return OllamaBackend()
    if name == "bedrock":
        return BedrockBackend()
    raise ValueError(f"Unknown backend: {name}")


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------
AGENT1_SYSTEM = """You are an Adversarial Clinical Data Extractor. You read unstructured
doctor notes and an insurance policy excerpt, and output a STRICT JSON object
matching the provided schema.

Rules:
- Be adversarial: only mark `criteria_fully_met = true` when EVERY payer
  criterion is explicitly supported by the doctor notes. When in doubt, false.
- Never invent ICD-10 / CPT codes. If absent, leave the list empty.
- Output JSON only. No commentary, no markdown fences.
"""

AGENT2A_SYSTEM = """You are a board-certified physician drafting a Medical Necessity
Justification Letter to an insurance payer. You are the ordering physician;
write in first person as the doctor.

Hard rules:
- Never mention AI, automation, language models, software, or that the letter
  was generated by a tool. You wrote it.
- Be highly concise: a tight one-page letter. No filler.
- Use formal clinical-legal register. Cite ICD-10 / CPT codes inline.
- Map the clinical evidence directly to each payer criterion.
- Close with the physician's name, NPI, and signature line.
"""

AGENT2B_SYSTEM = """You are a board-certified physician drafting a Peer-to-Peer Review
Request letter to an insurance payer's medical director. You are the ordering
physician; write in first person as the doctor.

Context: The standard prior authorization criteria are NOT fully met based on
documentation alone. You are NOT writing a justification letter. You are
requesting a clinical conversation with the payer's medical director.

Hard rules:
- Never mention AI, automation, language models, software, or that the letter
  was generated by a tool. You wrote it.
- Acknowledge the specific unmet criteria honestly and directly. Do NOT
  paper over gaps or fabricate evidence.
- Explain the clinical reasoning for why this patient still warrants the
  requested service despite the gap (e.g., individual patient factors,
  contraindications to alternatives, urgency).
- Explicitly request a peer-to-peer review with the medical director.
- Provide contact details placeholder and offer flexible scheduling.
- Be concise: one page maximum. Formal clinical-legal register.
- Close with the physician's name, NPI, and signature line.
"""


def _extract_json(raw: str) -> str:
    """Strip code fences / preamble that small local models sometimes emit."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    return m.group(0) if m else raw


# ---------------------------------------------------------------------------
# Agent functions (called by LangGraph nodes)
# ---------------------------------------------------------------------------
def run_agent_1(
    backend: InferenceBackend,
    doctor_notes: str,
    payer_guidelines: str,
) -> ExtractedClinicalCase:
    schema_json = json.dumps(ExtractedClinicalCase.model_json_schema(), indent=2)
    user_msg = f"""SCHEMA:
{schema_json}

DOCTOR NOTES:
\"\"\"{doctor_notes}\"\"\"

INSURANCE POLICY EXCERPT:
\"\"\"{payer_guidelines}\"\"\"

Return the JSON object now.
"""
    raw = backend.generate(AGENT1_SYSTEM, user_msg, json_mode=True)
    cleaned = _extract_json(raw)
    try:
        data = json.loads(cleaned)
        return ExtractedClinicalCase.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        logger.error("Agent 1 schema validation failed: %s", e)
        raise RuntimeError(
            "Agent 1 produced invalid JSON. This is the guardrail working - "
            "the pipeline refuses to hand bad data to the next agent.\n\n"
            f"Validation error: {e}\n\nRaw output (truncated):\n{raw[:800]}"
        ) from e


def _letter_user_msg(
    case: ExtractedClinicalCase,
    physician_name: str,
    physician_npi: str,
    patient_name: str,
    insurance_id: str,
    letter_kind: str,
) -> str:
    today = datetime.now().strftime("%B %d, %Y")
    return f"""CASE DATA (validated JSON from intake):
{case.model_dump_json(indent=2)}

LETTER PARAMETERS:
- Date: {today}
- Physician: {physician_name}
- NPI: {physician_npi}
- Patient: {patient_name}
- Insurance Member ID: {insurance_id}

Draft the {letter_kind} now. One page, formal, signed by the physician.
"""


def run_agent_2a_standard(backend, case, physician_name, physician_npi,
                          patient_name, insurance_id) -> str:
    user_msg = _letter_user_msg(case, physician_name, physician_npi,
                                patient_name, insurance_id,
                                "Medical Necessity Justification Letter")
    return backend.generate(
        AGENT2A_SYSTEM, user_msg, json_mode=False,
        model_override=getattr(backend, "writer_model", None),
    )


def run_agent_2b_peer_review(backend, case, physician_name, physician_npi,
                             patient_name, insurance_id) -> str:
    user_msg = _letter_user_msg(case, physician_name, physician_npi,
                                patient_name, insurance_id,
                                "Peer-to-Peer Review Request letter")
    return backend.generate(
        AGENT2B_SYSTEM, user_msg, json_mode=False,
        model_override=getattr(backend, "writer_model", None),
    )


# ===========================================================================
# LANGGRAPH ORCHESTRATION
# ===========================================================================

class PAState(TypedDict):
    """Shared whiteboard passed between nodes. Order = inputs first, outputs last."""
    # Inputs
    doctor_notes: str
    payer_guidelines: str
    physician_name: str
    physician_npi: str
    patient_name: str
    insurance_id: str
    backend_name: str  # "ollama" or "bedrock"
    # Outputs (populated by nodes)
    extracted_case: Optional[ExtractedClinicalCase]
    draft_letter: Optional[str]
    letter_type: Optional[Literal["standard", "peer_review"]]


def extractor_node(state: PAState) -> dict:
    """Agent 1: structured extraction with Pydantic guardrail."""
    backend = make_backend(state["backend_name"])
    case = run_agent_1(backend, state["doctor_notes"], state["payer_guidelines"])
    return {"extracted_case": case}


def standard_letter_node(state: PAState) -> dict:
    """Agent 2a: justification letter (criteria met path)."""
    backend = make_backend(state["backend_name"])
    letter = run_agent_2a_standard(
        backend, state["extracted_case"],
        state["physician_name"], state["physician_npi"],
        state["patient_name"], state["insurance_id"],
    )
    return {"draft_letter": letter, "letter_type": "standard"}


def peer_review_node(state: PAState) -> dict:
    """Agent 2b: peer-to-peer review request (criteria unmet path)."""
    backend = make_backend(state["backend_name"])
    letter = run_agent_2b_peer_review(
        backend, state["extracted_case"],
        state["physician_name"], state["physician_npi"],
        state["patient_name"], state["insurance_id"],
    )
    return {"draft_letter": letter, "letter_type": "peer_review"}


def route_after_extraction(state: PAState) -> Literal["standard", "peer_review"]:
    """Conditional edge: decide which letter agent based on extraction result."""
    if state["extracted_case"].criteria_fully_met:
        return "standard"
    return "peer_review"


@st.cache_resource
def build_graph():
    """
    Compile the LangGraph state machine once and cache it.
    @st.cache_resource ensures we don't rebuild on every Streamlit rerun.
    """
    graph = StateGraph(PAState)
    graph.add_node("extractor", extractor_node)
    graph.add_node("standard_letter", standard_letter_node)
    graph.add_node("peer_review", peer_review_node)

    graph.add_edge(START, "extractor")
    graph.add_conditional_edges(
        "extractor",
        route_after_extraction,
        {"standard": "standard_letter", "peer_review": "peer_review"},
    )
    graph.add_edge("standard_letter", END)
    graph.add_edge("peer_review", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------
def extract_pdf_text(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        if reader.is_encrypted:
            return "[ERROR] PDF is encrypted; please upload an unlocked copy."
        chunks = [p.extract_text() or "" for p in reader.pages]
        text = "\n\n".join(c for c in chunks if c.strip())
        return text if text else "[WARN] No extractable text (scanned PDF?)."
    except PdfReadError as e:
        return f"[ERROR] Could not parse PDF: {e}"
    except Exception as e:
        logger.exception("PDF extraction failed")
        return f"[ERROR] Unexpected PDF parsing failure: {e}"


# ===========================================================================
# STREAMLIT UI
# ===========================================================================

st.set_page_config(page_title="PA Multi-Agent Core", page_icon="🏥", layout="wide")

st.title("🏥 Healthcare Prior Authorization — Multi-Agent Core")
st.caption(
    "LangGraph state machine • Self-hosted dual-model inference • "
    "Pydantic-validated handoff • Editable HITL sign-off"
)

# ----- Sidebar -----
with st.sidebar:
    st.header("⚙️ Deployment Mode")
    mode = st.radio(
        "Inference backend",
        ["Local (Ollama)", "Cloud (AWS Bedrock)"],
        help="Local for dev/cost-free; Bedrock documented for production.",
    )
    backend_name = "ollama" if mode == "Local (Ollama)" else "bedrock"

    if backend_name == "ollama":
        st.info(
            f"Extractor: `{OLLAMA_EXTRACTOR_MODEL}`\n\n"
            f"Writer: `{OLLAMA_WRITER_MODEL}`\n\n"
            f"Endpoint: `{OLLAMA_URL}`"
        )
    else:
        st.info(f"Model: `{BEDROCK_MODEL_ID}`\nRegion: `{BEDROCK_REGION}`")

    st.divider()
    st.header("📋 Case Parameters")
    physician_name = st.text_input("Physician Name", value="Dr. Jane Doe, MD")
    physician_npi = st.text_input("NPI", value="1234567890")
    patient_name = st.text_input("Patient Name", value="John Smith")
    insurance_id = st.text_input("Insurance Member ID", value="ABC123456789")

# ----- Main inputs -----
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("📝 Doctor's Notes")
    doctor_notes = st.text_area(
        "Paste unstructured clinical notes",
        height=300,
        placeholder="Pt is a 54 y/o F with chronic lower back pain refractory to NSAIDs and 6 weeks PT...",
    )

with col_b:
    st.subheader("📑 Payer Guidelines")
    pdf_file = st.file_uploader("Upload policy PDF (optional)", type=["pdf"])
    guidelines_text = ""
    if pdf_file is not None:
        guidelines_text = extract_pdf_text(pdf_file)
        if guidelines_text.startswith(("[ERROR]", "[WARN]")):
            st.warning(guidelines_text)
            guidelines_text = ""
        else:
            st.success(f"Extracted {len(guidelines_text):,} chars from PDF.")
    payer_guidelines = st.text_area(
        "Or paste policy excerpt directly",
        value=guidelines_text,
        height=220,
    )

st.divider()

# ----- Session state -----
for key, default in [
    ("graph_result", None),
    ("edited_letter", None),
    ("letter_locked", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ----- Run pipeline -----
run_clicked = st.button(
    "🚀 Run Multi-Agent Pipeline",
    type="primary",
    use_container_width=True,
)

if run_clicked:
    if not doctor_notes.strip() or not payer_guidelines.strip():
        st.error("Both doctor's notes and payer guidelines are required.")
        st.stop()

    # Reset run state
    st.session_state.graph_result = None
    st.session_state.edited_letter = None
    st.session_state.letter_locked = False

    initial_state: PAState = {
        "doctor_notes": doctor_notes,
        "payer_guidelines": payer_guidelines,
        "physician_name": physician_name,
        "physician_npi": physician_npi,
        "patient_name": patient_name,
        "insurance_id": insurance_id,
        "backend_name": backend_name,
        "extracted_case": None,
        "draft_letter": None,
        "letter_type": None,
    }

    with st.status("Running LangGraph pipeline...", expanded=True) as status:
        try:
            graph = build_graph()
            st.write("📊 Graph compiled. Invoking...")
            final_state = graph.invoke(initial_state)
            st.session_state.graph_result = final_state
            st.session_state.edited_letter = final_state["draft_letter"]

            letter_kind = final_state["letter_type"]
            status.update(
                label=f"✅ Pipeline complete. Generated: {letter_kind.replace('_', ' ').title()}.",
                state="complete",
            )
        except RuntimeError as e:
            status.update(label="❌ Pipeline failed.", state="error")
            st.error(str(e))
            st.stop()

# ----- Display results -----
if st.session_state.graph_result is not None:
    result = st.session_state.graph_result
    case: ExtractedClinicalCase = result["extracted_case"]
    letter_type = result["letter_type"]

    # Route banner
    if letter_type == "standard":
        st.success(
            "✅ **Standard Justification Path** — "
            "All payer criteria are supported by the documentation."
        )
    else:
        st.warning(
            "⚠️ **Peer-to-Peer Review Path** — Some payer criteria are not "
            "fully met by the documentation. A peer-review request has been "
            "drafted instead of a justification letter, in accordance with "
            "compliance best practices."
        )
        if case.unmet_criteria:
            st.markdown("**Unmet criteria identified:**")
            for crit in case.unmet_criteria:
                st.markdown(f"- {crit}")

    with st.expander("🔍 Agent 1 — Structured Extraction", expanded=False):
        st.json(case.model_dump())

    # Editable letter
    st.subheader(f"📄 Draft: {letter_type.replace('_', ' ').title()} Letter")

    if not st.session_state.letter_locked:
        st.caption(
            "✏️ Edit the letter below before approving. Real clinical workflows "
            "always involve physician revision of generated drafts."
        )
        edited = st.text_area(
            "Letter (editable)",
            value=st.session_state.edited_letter,
            height=500,
            key="letter_editor",
            label_visibility="collapsed",
        )
        st.session_state.edited_letter = edited
    else:
        st.text_area(
            "Letter (locked)",
            value=st.session_state.edited_letter,
            height=500,
            disabled=True,
            label_visibility="collapsed",
        )

    st.divider()
    st.subheader("🛡️ Human-in-the-Loop Sign-Off")

    hitl_confirmed = st.checkbox(
        "I, the ordering physician, have personally reviewed and (where "
        "necessary) edited this letter, and verified all clinical claims, "
        "codes, and patient identifiers are accurate.",
        key="hitl_confirm",
        disabled=st.session_state.letter_locked,
    )

    if hitl_confirmed and not st.session_state.letter_locked:
        if st.button("🔒 Lock & Approve Letter", type="primary"):
            st.session_state.letter_locked = True
            st.rerun()

    if st.session_state.letter_locked:
        st.success("✅ Letter locked. Ready for export.")
        filename = (
            f"PA_{letter_type}_{patient_name.replace(' ', '_')}_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        st.download_button(
            "⬇️ Download Letter (.txt)",
            data=st.session_state.edited_letter,
            file_name=filename,
            mime="text/plain",
            use_container_width=True,
        )
