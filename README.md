# Healthcare Prior Authorization — Multi-Agent Core

> Personalize this opening paragraph in your own words before sharing it.
> Say why you built this and what you wanted to learn or prove. Two or
> three sentences is enough.

This is a small system that reads a doctor's clinical notes and an
insurance company's approval criteria, then writes one of two things: a
letter justifying the requested treatment, or a request for a
peer-to-peer review with the insurer's medical director if the
documentation doesn't fully support the request yet. It's built around two
AI agents working together rather than one model doing everything, and it
runs entirely on your own computer using a local model, with no cloud
costs and no API keys.

Prior authorization is one of the most time-consuming administrative tasks
in healthcare. Clinic staff spend hours reading payer policies and
matching them against patient charts before a doctor can even submit a
request. This project is a small proof of concept for automating the first
draft of that process, while keeping a doctor in the loop to check and
edit the final letter before anything goes out.

## How it works

The pipeline has two stages. First, an extraction agent reads the raw
notes and the payer's policy and turns them into a structured record:
diagnosis, codes, symptoms, what treatments have already been tried, and
whether every criterion in the payer's policy is actually backed up by the
notes. This step is validated against a strict schema (using Pydantic), so
if the model returns something malformed, the pipeline stops rather than
silently passing bad data forward.

Second, depending on whether that "criteria met" flag came back true or
false, a different writing agent takes over. If everything checks out, it
writes a standard justification letter. If something's missing, it writes
a peer-to-peer review request instead, openly naming the gap rather than
glossing over it. These are two separate prompts rather than one prompt
with a flag, because the two letters genuinely need to say different
things and ask for different outcomes.

The two stages and the branching logic between them are wired together
with LangGraph, which is built for exactly this kind of conditional,
multi-step agent workflow. Everything else in the project — the prompts,
the validation, the model-calling code — is written from scratch rather
than wrapped in a framework, because those parts didn't need one.

Once a letter is drafted, it shows up in an editable box in the
interface. A doctor (or in this demo, you) reviews it, edits anything that
needs fixing, ticks a box confirming they've checked it, and only then can
lock and download it. Nothing gets exported without that step.

## What's running under the hood

| Part | Tool | Reason |
|---|---|---|
| Agent orchestration | LangGraph | handles the conditional routing between agents |
| Data validation | Pydantic | enforces a strict schema between the two agent stages |
| Local inference | Ollama, running `llama3.2:3b` | free, self-hosted, runs on a normal laptop |
| Cloud inference (written, not active) | AWS Bedrock / Claude | documented as the production path, not used in this demo |
| Interface | Streamlit | quick to build, and its session state made the approval workflow straightforward |
| PDF reading | pypdf | lets you upload an actual policy PDF instead of pasting text |

The model-calling code is written as a small interface so swapping
providers (Ollama, Bedrock, or anything else) only means changing one
class, not rewriting the agents.

## Running it yourself

You'll need [Ollama](https://ollama.com) installed first.

```bash
ollama pull llama3.2:3b

conda create -n pa-agent python=3.11 -y
conda activate pa-agent

pip install -r requirements.txt

streamlit run app.py
```

Then open `http://localhost:8501` in your browser. Paste in some clinical
notes and a payer's policy (or upload a policy PDF), run the pipeline, and
you'll see the extracted data, the routing decision, and the draft letter.

A note on hardware: this was built and tested on an 8GB Apple Silicon Mac,
which is enough to run the 3B model comfortably. Anything larger (8B and
up) writes noticeably better prose but needs more memory than an 8GB
machine reliably has to spare.

## What testing actually showed

I tested this with real model calls rather than stubbing anything out, and
it surfaced a genuine limitation worth being upfront about. When a payer
criterion is phrased as an absence — something like "no history of
pancreatitis" — and the clinical notes explicitly confirm that absence,
the small local model consistently misreads this as the criterion not
being met. I tried fixing this twice with more specific prompt
instructions, and the same mistake showed up a third time in the same
form. It's a real limit of using a 3-billion-parameter model for this kind
of careful, multi-step reading, not something I could prompt my way out
of.

The reason this doesn't undermine the system is that the extraction agent
is deliberately built to be cautious: it's instructed to default to
"criteria not met" whenever it isn't fully certain. So this particular
mistake only ever pushes the output toward the safer outcome, a
peer-to-peer review request instead of a justification letter. In every
test I ran, the system never went the other way and claimed criteria were
met when they weren't. A bigger model would likely get this right, and the
model used for each stage is configurable via an environment variable for
that reason.

## What I'd add with more time

- A small rules-based check for absence-type and numeric criteria, run
  before the LLM extraction step, to catch the cases above
- A Dockerfile so this can be deployed the same way anywhere
- A labeled test set so model accuracy could actually be measured rather
  than spot-checked
- Basic tracing/logging of each agent call for debugging at scale

## Files

```
app.py            the whole application: agents, the graph, the interface
requirements.txt  pinned dependencies
```

## About

> Add your name, a link to your LinkedIn or portfolio, and a line about
> your background here.
