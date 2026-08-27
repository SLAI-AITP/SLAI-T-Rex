You are improving prompts for an OR-CPT data factory.

Return exactly one JSON object with these keys:
- `target_prompt`: one of `backtranslation_prompt` or `forward_modeling_prompt`.
- `revised_prompt`: the complete replacement prompt text.
- `change_summary`: concise explanation of what changed and why.
- `risk_notes`: possible regressions or cases that should be watched after promotion.

Optimization policy:
- Preserve the original task boundary of the target prompt.
- Do not add hidden chain-of-thought requirements.
- Do not ask the model to reveal solver truth, reference objective, or validation metadata in natural-language problem statements.
- Prefer precise, testable instructions over broad advice.
- Keep the prompt in English because generated CPT data is English.
- The revised prompt must be complete; do not output a patch or diff.

Current run diagnostics:

```json
{{RUN_DIAGNOSTICS_JSON}}
```

Current target prompt:

```text
{{CURRENT_PROMPT}}
```
