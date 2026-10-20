"""TrustFin runtime layer (Phase-5A skeleton — roadmap docs/next_phases_roadmap.md §4/§8).

Model-free, deterministic, offline-testable code that implements the response
contract for a verified financial assistant:

  runtime.response   JSON-Schema validation of the structured response (§8)
  runtime.answer_render  render the displayed number from the executor result
  runtime.redaction  PII detection/redaction (privacy gate)
  runtime.policy     input routing + risk gates + frozen-threshold abstention

The SLM, retrieval and calculator tools are wired in behind this contract in
Phase 6/8; nothing here calls a model or the network.
"""
