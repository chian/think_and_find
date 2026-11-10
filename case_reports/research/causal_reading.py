"""Source extraction and validator feedback within the existing reading episode."""
from dataclasses import replace
from copy import deepcopy

from .documents import Span
from .records import Ledger, ResearchResult, identity
from .engine import selected_choice
from .support_observations import observation_identity, observations_schema, validate_observation_fields
from .source_claims import claim_identity, claims_schema, validate_claim_fields
from . import source_anchors
from ...task_executors._http import ProviderError
from ...task_executors.evidence_extraction.argo import ModelOutputError


def _repair_contract(errors, extraction_schema):
    """Only host validator targets grant permission to change the base response."""
    paths = []
    for error in errors:
        if "path_segments" in error and error["path_segments"] not in paths:
            paths.append(error["path_segments"])
    paths = [path for path in paths if not any(
        len(ancestor) < len(path) and path[:len(ancestor)] == ancestor
        for ancestor in paths)]
    issues, choices = [], []
    for index, path in enumerate(paths):
        schema = extraction_schema
        for segment in path:
            if type(segment) is int:
                schema = schema["items"]
            else:
                properties = schema.get("properties", {})
                schema = properties[segment] if segment in properties else schema["additionalProperties"]
        removable = (len(path) == 5 and path[0] == "findings"
                     and path[2] == "measurements" and path[3] in {"claims", "observations"})
        replacement_schema = {"anyOf": [schema, {"type": "null"}]} if removable else schema
        issues.append({"issue": index, "path_segments": list(path),
                       "removable": removable, "replacement_schema": replacement_schema})
        choices.append({"type": "object", "additionalProperties": False,
            "required": ["issue", "replacement"], "properties": {
                "issue": {"const": index}, "replacement": replacement_schema}})
    if not issues:
        raise ValueError("Extraction repair requires a host validator target")
    return issues, {"type": "object", "additionalProperties": False,
        "required": ["corrections"], "properties": {
            "corrections": {"type": "array", "items": {"oneOf": choices}}}}


def _apply_corrections(base, response, issues):
    """Patch rejected targets; accepted siblings never enter the editable scope."""
    from jsonschema import Draft202012Validator

    if (not isinstance(response, dict) or set(response) != {"corrections"}
            or not isinstance(response["corrections"], list)):
        raise ValueError("Extraction repair requires exactly a corrections array")
    requested = {issue["issue"]: issue for issue in issues}
    supplied = {}
    for correction in response["corrections"]:
        if not isinstance(correction, dict) or set(correction) != {"issue", "replacement"}:
            raise ValueError("Each correction requires exactly issue and replacement")
        index = correction["issue"]
        if type(index) is not int or index not in requested:
            raise ValueError(f"Unknown extraction repair issue: {index!r}")
        if index in supplied:
            raise ValueError(f"Duplicate extraction repair issue: {index}")
        supplied[index] = correction["replacement"]
    if supplied.keys() != requested.keys():
        raise ValueError(f"Missing extraction repair issues: {sorted(requested.keys() - supplied.keys())}")
    patched, feedback = deepcopy(base), []
    for index, issue in requested.items():
        replacement = supplied[index]
        invalid = list(Draft202012Validator(issue["replacement_schema"]).iter_errors(replacement))
        if invalid:
            # Keep this target rejected while allowing other corrections to be
            # admitted. Feedback has no path_segments and grants no edit rights.
            feedback.extend({"path": f"corrections[{index}].replacement",
                "target_path_segments": list(issue["path_segments"]),
                "why": f"{error.json_path}: {error.message}", "rejected": deepcopy(replacement)}
                for error in invalid)
            continue
        path = issue["path_segments"]
        if not path:
            patched = deepcopy(replacement)
            continue
        parent = patched
        for segment in path[:-1]:
            parent = parent[segment]
        if replacement is None and issue["removable"]:
            del parent[path[-1]]
        else:
            parent[path[-1]] = deepcopy(replacement)
    return patched, feedback


def read_source_window(reader, spans, question):
    from .discovery_goals import reader_extraction_context

    allowed = {}
    for span in spans:
        for offset in range(0, len(span.text), 800):
            text = span.text[offset:offset + 800]
            key = span.key if len(span.text) <= 800 else f"{span.key}:part:{offset}"
            allowed[key] = Span(key, span.start + offset, span.start + offset + len(text), text)
    allowed.update({anchor["span"]: reader.seen_spans[anchor["span"]] for anchor in reader.memory["anchors"]})
    first_call = reader.run.model_calls + 1
    anchor_context = source_anchors.context(reader, allowed)
    fields = {"span": {"enum": list(allowed)},
        "study_id": {"type": "string", "minLength": 1},
        "finding": {"type": "string", "minLength": 1},
        "measurements": {"type": "object", "additionalProperties": False,
            "required": ["observations", "claims"],
            "properties": {"observations": observations_schema(source=reader.document.source),
                           "claims": claims_schema(source=reader.document.source)}},
        "source_anchors": source_anchors.input_schema(allowed, anchor_context["units"])}
    memory_schema = {"type": "object", "additionalProperties": False,
        "required": ["summary", "unresolved_links", "anchors", "retained_evidence"], "properties": {
            "summary": {"type": "string"},
            "unresolved_links": {"type": "array", "items": {"type": "string"}},
            "retained_evidence": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["key", "version"], "properties": {"key": {"type": "string"}, "version": {"type": "string"}}}},
            "anchors": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["span"], "properties": {"span": {"enum": list(allowed)}}}}}}
    response, returned, errors, correction_feedback = None, None, [], []
    previous = reader.extraction_repair
    reader.extraction_repair = None
    try:
        request = {"operation": "extract_causal_evidence", "source": reader.document.source,
            "source_version": reader.document.sha256, "question": question,
            "task": reader.instructions,
            "extraction_request": reader_extraction_context(reader, question),
            "passages": source_anchors.passages(reader, allowed),
            "source_anchor_context": anchor_context,
            "reading_memory": reader.memory,
            "retained_records": [reader.ledger.causal_sheet.snapshot(ref)
                                 for ref in reader.memory["retained_evidence"]],
            "instruction": (
                "Extract source information answering the reading question. Return findings, native measurements, "
                "source claims, exact span references and source-specific qualifications. Page aggregates the "
                "source answer; Scientific owns theory reconstruction. Do not formulate cross-study arguments, "
                "research scopes or grading needs. Use extraction_request for the acquisition information need "
                "and native claim/observation definitions. Preserve source-native meanings and vocabulary. "
                "This request is context, not source evidence. Return source observations and claims with exact "
                "provenance; Discovery assembles its answer and Scientific reconstructs the theory. "
                "Distinguish measured, simulated and proposed results. Do not invent values, source identifiers, "
                "equations, units or uncertainty. Return no findings when the passage supplies no requested information. "
                "Put bounded absence-of-information reports, uninspected-source warnings and unrelated citation or "
                "recommendation leads in reading_memory, not findings. Return findings=[] when that is all the "
                "passage supplies. Source-reported negative results, definitions, assumptions, limitations and "
                "requested provenance of the retrieved work remain source information; anchor only what answers "
                "the reading question. "
                "study_id identifies the source study; use its URL when studies cannot be distinguished. Reuse source-native "
                "identifiers and precise locators across repeated extractions. Preserve native table headings and units. "
                "Observation source_id is optional: use the source's own full observation ID, or an empty string plus "
                "a precise locator for that quantity and condition. Never put the document URL or study ID there. "
                "Each finding includes source_anchors maps matching its native observation/claim selectors. "
                "Use exact quotes from the unit-labeled passage fragments. A statement quote lies inside one host "
                "unit. Observations also require result={span,unit,quote,occurrence}: quote only the exact reported "
                "value, without units, uncertainty or context, and select its 1-based occurrence within the selected "
                "unit's supplied portion. Distinct equal-valued cells select their own source positions. Text values "
                "copy the source wording; Boolean values require literal true/false source values. Anchor the precise quantity/subject and source conditions separately; avoid broad quotes "
                "containing multiple quantities or conditions. Native fields retain the source meanings; anchor "
                "sidecars supply source-occurrence identity when the source has no native ID. Locator prose and "
                "paraphrasing cannot create a new occurrence. Prose-only findings need finding anchors. "
                "Separate reported results require their own result occurrences and quantity anchors. Adding or "
                "dropping descriptive condition clauses does not make an existing result a new observation. "
                "Each observation declares kind: measurement, fit_parameter, simulation_input, simulation_output, "
                "model_prediction, reported_property or unknown. A fitted saturation rate is fit_parameter, never "
                "an observed flux at a finite substrate concentration. Use the commissioning parent's defined "
                "quantity vocabulary and condition keys/units when the source establishes that correspondence; "
                "retain original wording in exact source anchors. Unknown correspondence must remain unknown. "
                "Conditions may be text; structured condition maps require "
                "Python-identifier names and numeric or rational values. Put categorical conditions in text. "
                "Use not reported for absent uncertainty; reported intervals use kind=interval, lower and upper. "
                "Claims use kind=equation with relation = and defined symbols; causal_relation uses causes, increases, "
                "decreases, inhibits, enables, prevents, has_no_effect_on, reverses_effect_on, depends_on, mediates, "
                "moderates or affects; boundary_condition uses applies_when, fails_when, changes_when or requires. "
                "Each claim's statement_id is the source's own equation or statement label, such as '(1)'; "
                "use an empty string when the source gives no statement label. Never put a document URL, study ID "
                "or invented label in statement_id. Reuse the same native statement label and precise locator "
                "across windows; separate unlabeled claims in one paragraph retain their own relations and symbols. "
                "Non-equation claims require symbols={}. Empty observations/claims maps are appropriate when absent. "
                "Local extraction credit measures newly returned source information; repeating it adds zero. "
                "reading_memory summarizes the extracted source information and unresolved source links. Retain exact "
                "previously supplied source-record references needed for that answer. Choose anchors from supplied spans. "
                "When repair is supplied, return targeted corrections for its host-indexed issues only. "
                "Include each issue exactly once with a replacement matching its declared schema. Accepted fields "
                "cannot be edited: do not reword accepted findings, claims, locators, conditions or memory. "
                "Null is permitted only for a rejected claim or observation that cannot be represented; its finding "
                "prose remains. Whole-response replacement is permitted only for an issue whose path_segments is []."),
            "response_contract": {"type": "object", "additionalProperties": False,
                "required": ["findings", "reading_memory"], "properties": {
                    "findings": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                        "required": list(fields), "properties": fields}}, "reading_memory": memory_schema}}}
        if previous is not None:
            issues, contract = _repair_contract(previous["errors"], request["response_contract"])
            request["repair"] = {"base_response": previous["response"], "errors": previous["errors"],
                                 "issues": issues}
            request["response_contract"] = contract
        returned = reader.run.complete(request, tier=reader.extraction_tier)
        if previous is None:
            response = returned
        else:
            response, correction_feedback = _apply_corrections(previous["response"], returned, issues)
        if not isinstance(response, dict) or set(response) != {"findings", "reading_memory"}:
            errors.append({"path": "response", "path_segments": [],
                "why": "Extraction requires exactly findings and reading_memory", "rejected": deepcopy(response)})
            raise ValueError("Extraction requires exactly findings and reading_memory")
        findings = response["findings"]
        if not isinstance(findings, list):
            errors.append({"path": "response", "path_segments": [],
                "why": "findings must be an array", "rejected": deepcopy(response)})
            raise ValueError("findings must be an array")
        working = Ledger(reader.ledger.result())
        updates, diagnostics = [], []
        known = source_anchors.registry(reader)
        for index, item in enumerate(findings):
            try:
                if (not isinstance(item, dict) or set(item) - set(fields)
                        or set(fields) - set(item) - {"source_anchors"}):
                    raise ValueError("Finding requires exactly span, study_id, finding, measurements and source_anchors")
                span = allowed[selected_choice(item["span"], allowed, "source span")]
                native = item["measurements"]
                if not isinstance(native, dict) or set(native) != {"observations", "claims"}:
                    raise ValueError("measurements requires exactly observations and claims")
                native = {name: dict(value) if isinstance(value, dict) else value for name, value in native.items()}
                data = {"source": reader.document.source, "locator": f"characters {span.start}:{span.end}",
                    "study_id": item["study_id"], "finding": item["finding"],
                    "measurements": {"native": native, "quote": span.text,
                        "source_version": reader.document.sha256, "question": question}}
                sidecar = {"observations": {}, "claims": {}, "finding": [], "pending": False}
                raw_anchors = item.get("source_anchors")
                if not isinstance(raw_anchors, dict) or set(raw_anchors) != {"observations", "claims", "finding"}:
                    errors.append({"path": f"findings[{index}].source_anchors",
                        "path_segments": ["findings", index, "source_anchors"],
                        "why": "Source anchors require exactly observations, claims and finding maps/array",
                        "rejected": deepcopy(raw_anchors)})
                    raw_anchors = {"observations": {}, "claims": {}, "finding": []}
                    sidecar["pending"] = True
                trial_known = deepcopy(known)
                for name, validate in (("observations", validate_observation_fields), ("claims", validate_claim_fields)):
                    if not isinstance(native[name], dict):
                        raise ValueError(f"{name} must be a named map")
                    rejected = []
                    for selector in list(native[name]):
                        try:
                            validate(data, selector)
                        except (ValueError, TypeError, KeyError) as exc:
                            errors.append({"path": f"findings[{index}].measurements.{name}.{selector}",
                                "path_segments": ["findings", index, "measurements", name, selector],
                                "why": str(exc), "rejected": deepcopy(native[name][selector])})
                            rejected.append(selector)
                    for selector in rejected:
                        del native[name][selector]
                    pending = []
                    for selector in list(native[name]):
                        try:
                            supplied = raw_anchors[name]
                            if not isinstance(supplied, dict):
                                raise source_anchors.SourceAnchorError([name], "Source anchors must be a selector map")
                            entry_known = deepcopy(trial_known)
                            entry = source_anchors.resolve_entry(reader, allowed, name, selector,
                                supplied.get(selector), entry_known)
                            if name == "observations":
                                try:
                                    source_anchors.validate_result_value(native[name][selector]["value"], entry["result"])
                                except ValueError as exc:
                                    errors.append({"path": f"findings[{index}].measurements.{name}.{selector}",
                                        "path_segments": ["findings", index, "measurements", name, selector],
                                        "why": str(exc), "rejected": deepcopy(native[name][selector])})
                                    pending.append(selector)
                                    continue
                            sidecar[name][selector] = entry
                            trial_known = entry_known
                        except source_anchors.SourceAnchorError as exc:
                            errors.append({"path": f"findings[{index}].source_anchors." + ".".join(map(str, exc.path)),
                                "path_segments": ["findings", index, "source_anchors", *exc.path],
                                "why": str(exc), "rejected": deepcopy(raw_anchors.get(name))})
                            pending.append(selector)
                    for selector in pending:
                        del native[name][selector]
                    sidecar["pending"] |= bool(pending or rejected)
                try:
                    sidecar["finding"] = source_anchors.resolve_finding(reader, allowed, raw_anchors["finding"], trial_known)
                    if item["finding"].strip() and not any(native.values()) and not sidecar["finding"] and not sidecar["pending"]:
                        raise source_anchors.SourceAnchorError(["finding"], "A prose-only finding needs an exact source unit anchor")
                except source_anchors.SourceAnchorError as exc:
                    errors.append({"path": f"findings[{index}].source_anchors." + ".".join(map(str, exc.path)),
                        "path_segments": ["findings", index, "source_anchors", *exc.path],
                        "why": str(exc), "rejected": deepcopy(raw_anchors.get("finding"))})
                    sidecar["pending"] = True
                data["measurements"]["source_anchors"] = source_anchors.finalize(reader, allowed, sidecar)
                # Two differently described quantities/conditions assigned the same
                # no-ID occurrence in this response reveal ambiguous sidecars, not
                # permission to rewrite accepted scientific measurement fields.
                seen, collisions = {}, set()
                for selector, observation in native["observations"].items():
                    if observation["source_id"].strip():
                        continue
                    ident = observation_identity(data, selector)[0]
                    scope = (" ".join(observation["quantity"].split()), observation["conditions"])
                    if ident in seen and seen[ident][1] != scope:
                        collisions.update((selector, seen[ident][0]))
                    else:
                        seen[ident] = (selector, scope)
                for selector in collisions:
                    errors.append({"path": f"findings[{index}].source_anchors.observations.{selector}",
                        "path_segments": ["findings", index, "source_anchors", "observations", selector],
                        "why": "Different native quantities/conditions share one anchored result. Select the actual distinct result occurrence and quantity; descriptive condition changes cannot create another observation. Accepted native fields remain fixed.",
                        "rejected": deepcopy(raw_anchors["observations"].get(selector))})
                    del native["observations"][selector]
                    del sidecar["observations"][selector]
                    sidecar["pending"] = True
                data["measurements"]["source_anchors"] = source_anchors.finalize(reader, allowed, sidecar)
                source_anchors.retain_roles(known, sidecar)
                key = "evidence:" + identity([reader.document.source, reader.document.sha256,
                    span.start, span.text, item["study_id"], item["finding"]])[:24]
                updates.append(working.causal_sheet.put(key, "evidence", data))
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                errors.append({"path": f"findings[{index}]", "path_segments": ["findings", index],
                    "why": str(exc), "rejected": item})
        try:
            proposed = response["reading_memory"]
            memory = {**proposed, "anchors": [{"span": anchor["span"], "quote": allowed[anchor["span"]].text}
                                               for anchor in proposed["anchors"]]}
            reader.validate_memory(memory, allowed)
            reader.pending_memory = memory
            reader.pending_memory_update = True
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            errors.append({"path": "reading_memory", "path_segments": ["reading_memory"],
                "why": str(exc), "rejected": response["reading_memory"]})
        errors.extend(correction_feedback)
        diagnostics.extend({"stage": "source_extraction", "status": "rejected", "category": "invalid_output",
            "repair_owner": "originating_reader", "repair_action": "correct_rejected_output", **error} for error in errors)
        result = ResearchResult(causal_sheet_updates=tuple(updates), diagnostics=tuple(diagnostics),
            evaluated=bool(updates) or not any("path_segments" in error for error in errors))
    except (ProviderError, ValueError, TypeError, KeyError, OSError) as exc:
        result = reader.failure_result(exc, "source_reading")
        if previous is not None and response is None:
            # API, JSON and correction-envelope failures cannot reopen accepted
            # fields or replace the base with an invalid correction response.
            response = deepcopy(previous["response"])
            errors = deepcopy(previous["errors"])
            errors.append({"path": "repair_response", "why": str(exc), "rejected":
                deepcopy(exc.raw_response if isinstance(exc, ModelOutputError) else returned)})
        elif not errors and previous is not None:
            response = deepcopy(previous["response"])
            errors = deepcopy(previous["errors"])
            errors.append({"path": "repair_response", "why": str(exc), "rejected": deepcopy(returned)})
        elif not errors and response is None:
            response = exc.raw_response if isinstance(exc, ModelOutputError) else response
            errors.append({"path": "response", "path_segments": [],
                           "why": str(exc), "rejected": deepcopy(response)})
        elif not errors:
            # An internal failure after a well-shaped initial response is not
            # permission for the model to rewrite an otherwise accepted scope.
            raise
        errors.extend(correction_feedback)
    unresolved = any("path_segments" in error for error in errors)
    if unresolved:
        reader.extraction_repair = {"spans": spans, "question": question, "response": response, "errors": errors}
    reader.pending_spans = spans
    reader.seen_spans.update(allowed)
    memory = reader.pending_memory or reader.memory
    attempt = {"source": reader.document.source, "source_version": reader.document.sha256,
        "document_characters": len(reader.document.text), "episode_id": reader.goal.goal_id,
        "first_model_call": first_call, "question": question, "page_task": getattr(reader, "page_task", None),
        "ranges": [[s.start, s.end] for s in spans],
        "status": ("partial" if unresolved else "completed") if result.evaluated else "failed",
        "reading_summary": memory["summary"], "unresolved_links": memory["unresolved_links"]}
    return replace(result, reading_attempts=(attempt,))
