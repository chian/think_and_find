"""Shared theory requests, typed additions, and request-relative contributions.

Episode decisions remain in their sources. This module connects the same theory
language to commissions, source evidence, stored fragments, and local credit.
"""
from copy import deepcopy

from .records import Credit, Ledger, ResearchResult


COMPONENT_GROUPS = ("entities", "quantities", "statements", "gaps")
COMPONENT_KINDS = ("entity", "quantity", "equation", "causal_relation", "mechanism",
                   "assumption", "derivation", "approximation", "gap")


def request_schema():
    return {"type": "object", "additionalProperties": False,
        "required": ["reconstruction", "focus", "requested_kinds"], "properties": {
            "reconstruction": {"type": "string", "minLength": 1},
            "focus": {"type": "array", "uniqueItems": True,
                      "items": {"type": "string", "minLength": 1}},
            "requested_kinds": {"type": "array", "minItems": 1, "uniqueItems": True,
                                "items": {"enum": list(COMPONENT_KINDS)}}}}


def validate_request(request, sheet=None):
    from jsonschema import Draft202012Validator
    errors = list(Draft202012Validator(request_schema()).iter_errors(request))
    if errors:
        error = errors[0]
        raise ValueError(f"Theory request {error.json_path}: {error.message}")
    if sheet is not None:
        fragment = fragment_from_sheet(sheet, request["reconstruction"])
        ids = {key for group in COMPONENT_GROUPS for key in fragment[group]}
        if not set(request["focus"]) <= ids:
            raise ValueError("Theory request focus must identify declared components")
    return deepcopy(request)


def fragment_from_sheet(sheet, key):
    sheet._require_kind(key, "theory_fragment")
    return deepcopy(sheet.rows[key]["data"])


def evidence_lookup(sheet):
    def lookup(reference):
        row = sheet.snapshot(reference)
        if row["kind"] != "evidence":
            raise ValueError("Theory components must cite exact source evidence versions")
        return row["data"]
    return lookup


def selected_source_evidence(source):
    """Resolve commissioned, selected and repair-specific evidence, not history."""
    sheet = source.ledger.causal_sheet
    refs = {}

    def select(ref):
        row = sheet.snapshot(ref)
        if row["kind"] == "evidence":
            refs[(row["key"], row["version"])] = ref

    for key in dict.fromkeys([*source.commission.inputs, *source.selected_keys]):
        select(sheet.anchor(key))
    for ref in [*source.scoped_snapshots(), *source.working_assessment["record_refs"]]:
        select(ref)

    # Format repair needs the actual evidence cited by the rejected additions.
    # Invalid references stay in validator feedback; known ones supply their
    # exact source content without copying every acquired source record.
    def repair_references(value):
        if isinstance(value, dict):
            evidence = value.get("evidence", ())
            for ref in evidence if isinstance(evidence, (list, tuple)) else ():
                if (isinstance(ref, dict) and set(ref) == {"key", "version"}
                        and isinstance(ref["key"], str) and isinstance(ref["version"], str)
                        and (ref["key"], ref["version"]) in known):
                    select(ref)
            for child in value.values():
                repair_references(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                repair_references(child)

    known = {(row["key"], row["version"]) for row in sheet.history if row["kind"] == "evidence"}
    repair_references(getattr(source, "theory_repair", None))
    return sheet.snapshot_view(list(refs.values()))


def theory_context(source):
    from .theory_language import language_reference, normalization_report
    sheet = source.ledger.causal_sheet
    request = validate_request(source.commission.theory_request, sheet)
    fragment = fragment_from_sheet(sheet, request["reconstruction"])
    from .theory_language import evidence_references
    refs = {(ref["key"], ref["version"]): ref for ref in evidence_references(fragment)}
    selected = selected_source_evidence(source)
    for key, row in sheet.of_kind("evidence"):
        refs[(key, row["version"])] = {"key": key, "version": row["version"]}
    for row in selected:
        refs[(row["key"], row["version"])] = {"key": row["key"], "version": row["version"]}
    available = []
    for ref in refs.values():
        row = sheet.snapshot(ref)
        available.append({**ref, "kind": "evidence", "source": row["data"]["source"],
                          "locator": row["data"]["locator"]})
    return {"theory_request": request, "theory_fragment": fragment,
            "source_evidence": selected, "available_source_evidence": available,
            "theory_language": language_reference(),
            "formal_comparisons": normalization_report(fragment, evidence_lookup=evidence_lookup(sheet))}


def deduplicate_theory_inputs(source, inputs):
    """Present each selected record body once across literature context fields."""
    inputs = dict(inputs)
    key = source.commission.theory_request["reconstruction"]
    fragment = source.ledger.causal_sheet.rows[key]
    shown = {(key, fragment["version"]): "theory_fragment"}
    for field in ("source_evidence", "scientific_inputs", "historical_inputs"):
        if field not in inputs:
            continue
        records = []
        for row in inputs[field]:
            ref = (row["key"], row["version"])
            if ref in shown:
                records.append({**{name: value for name, value in row.items() if name != "data"},
                                "content_in": shown[ref]})
            else:
                records.append(row)
                shown[ref] = field
        inputs[field] = records
    return inputs


def reader_theory_context(source):
    """Pass extraction meanings to processors, not the scientific parent's evidence."""
    request = validate_request(source.commission.theory_request, source.ledger.causal_sheet)
    fragment = fragment_from_sheet(source.ledger.causal_sheet, request["reconstruction"])
    def definition(component):
        return {key: deepcopy(value) for key, value in component.items()
                if key not in {"evidence", "supersedes"}}
    focus = {key: definition(component) for group in ("statements", "gaps")
             for key, component in fragment[group].items() if key in request["focus"]}
    return {"request": request,
            "entities": {key: definition(value) for key, value in fragment["entities"].items()},
            "quantities": {key: definition(value) for key, value in fragment["quantities"].items()},
            "focused_components": focus,
            "instruction": "Extract the source's definitions, relationships, mechanisms, assumptions and "
                "derivations relevant to this question. Use supplied meanings for already bound quantities. "
                "Report additional source-defined quantities explicitly in findings and claim symbols. "
                "Distinguish an unspecified relationship from a missing passage. Page constructs the typed "
                "theory fragment from these source records; this reader returns source extraction only."}


def integration_schema():
    from .theory_language import additions_schema
    return {"type": "object", "additionalProperties": False,
        "required": ["action", "additions", "rationale"], "properties": {
            "action": {"const": "integrate_theory"}, "additions": additions_schema(),
            "rationale": {"type": "string", "minLength": 1}}}


def integrate_theory(source, additions):
    """Validate a proposed graph addition and retain a rejected payload for repair."""
    from .theory_language import apply_patch
    from .causal_sheet import CausalSheet
    sheet = source.ledger.causal_sheet
    request = validate_request(source.commission.theory_request, sheet)
    try:
        fragment = apply_patch(fragment_from_sheet(sheet, request["reconstruction"]), additions,
                               evidence_lookup=evidence_lookup(sheet))
        staged = CausalSheet(sheet.history)
        row = staged.put(request["reconstruction"], "theory_fragment", fragment)
    except (ValueError, TypeError, KeyError) as exc:
        rejected = {"additions": deepcopy(additions), "errors": [str(exc)]}
        source.theory_repair = rejected
        return ResearchResult(evaluated=False, diagnostics=({
            "stage": "theory_integration", "status": "rejected", "category": "invalid_output",
            "why": str(exc), "rejected": deepcopy(additions),
            "repair_owner": "theory_format_repair", "repair_action": "correct_the_typed_additions"},))
    source.theory_repair = None
    return ResearchResult(causal_sheet_updates=(row,))


def component_index(fragment):
    from .theory_language import component_kind
    return {key: (component_kind(component, group), component)
            for group in COMPONENT_GROUPS for key, component in fragment[group].items()}


def _referenced_ids(component):
    """Typed references and expression quantity bindings connect the shared graph."""
    from .theory_language import collect_component_refs
    refs = collect_component_refs(component)
    return {ref["id"] if isinstance(ref, dict) else ref for ref in refs}


def _requested_components(fragment, request):
    index = component_index(fragment)
    focus = set(request["focus"])
    # A gap question selects the components that generated it, not an arbitrary
    # equation-shaped blank. New statements can answer through those bindings.
    for key in tuple(focus):
        if key in fragment["gaps"]:
            focus.update(_referenced_ids(fragment["gaps"][key]))
    def related(key):
        if not focus:
            return True
        todo, seen = [key], set()
        while todo:
            current = todo.pop()
            if current in focus:
                return True
            if current in seen or current not in index:
                continue
            seen.add(current)
            todo.extend(_referenced_ids(index[current][1]))
        return False
    return {key for key, (kind, _) in index.items()
            if kind in request["requested_kinds"] and related(key)}


def _contributions(sheet, request):
    from .theory_language import contribution_identities
    key = request["reconstruction"]
    if key not in sheet.rows:
        return set()
    fragment = fragment_from_sheet(sheet, key)
    identities = contribution_identities(fragment, evidence_lookup=evidence_lookup(sheet))
    allowed = _requested_components(fragment, request)
    index = component_index(fragment)
    # Declaring a question is useful steering, but unanswered gaps and parent
    # vocabulary seeds do not count as a fulfilled literature contribution.
    return {ident for component, values in identities.items() if component in allowed
            and index[component][1]["evidence"]
            for ident in values}


def theory_credit(result, baseline, request, column):
    returned = _contributions(Ledger(result).causal_sheet, request)
    inherited = _contributions(Ledger(baseline).causal_sheet, request)
    return Credit(evaluated=result.evaluated, quality_resolution=1.0,
        qualities=tuple((column, key, 1.0) for key in sorted(returned - inherited)))


def theory_report(source, result):
    sheet = Ledger(result).causal_sheet
    request = source.commission.theory_request
    fragment = fragment_from_sheet(sheet, request["reconstruction"])
    row = sheet.rows[request["reconstruction"]]
    previous = Ledger(source.initial).causal_sheet
    before = (fragment_from_sheet(previous, request["reconstruction"])
              if request["reconstruction"] in previous.rows else {group: {} for group in COMPONENT_GROUPS})
    additions = {group: [key for key in fragment[group] if key not in before[group]]
                 for group in COMPONENT_GROUPS}
    return {"theory_fragment": {"key": row["key"], "version": row["version"]},
            "theory_contributions": {"request": deepcopy(request), "added_components": additions,
                "identities": sorted(_contributions(sheet, request) - _contributions(previous, request)),
                "repair": deepcopy(getattr(source, "theory_repair", None))}}
