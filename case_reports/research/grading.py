"""Dispatch and bounded views for executable scientific subscoring sheets.

These are calculator operations, not episode roles or model verdicts.
"""
from importlib import import_module

GRADING_FIELDS = {
    "expectation_test": ("expectation", "assessment", "scope", "claim_relation", "claim_intervention", "target", "grounds", "rationale"),
    "repair_assessment": ("break", "expectation_test", "target", "target_test", "candidate", "alternatives", "claim_relation", "construction_cases", "test_cases", "rationale", "limitations"),
    "support_assessment": ("support", "application", "importance", "bounds_test", "source_formalization", "prediction", "facts", "limitations"),
}
MODULES = {"expectation_test": "bounds", "repair_assessment": "repairs", "support_assessment": "support_grading"}
CALCULATORS = {"expectation_test": "grade_bounds", "repair_assessment": "grade_repair", "support_assessment": "grade_support"}


def module_for(kind):
    return import_module(f"{__package__}.{MODULES[kind]}")


def grade_record(sheet, key):
    kind = sheet.rows[key]["kind"]
    return getattr(module_for(kind), CALCULATORS[kind])(sheet, key)


def compact_calculation(value, *, offset=0, limit=8, summary=False):
    """Page verbose case/alternative/observation details without hiding totals.

    Full calculator results stay in host state. Each truncated field reports its
    total and next offset; these offsets are only for presentation, never domain
    membership or scientific calculation.
    """
    if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError("Calculation detail paging requires nonnegative offset and limit in 1..8")
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key == "source" and isinstance(item, dict) and "measurement" in item:
                # Exact source operands already have an evidence/observation key
                # beside this summary; don't copy their whole extraction again.
                result[key] = {name: item[name] for name in ("study_id", "source", "evidence_locator", "observation", "source_version") if name in item}
                result[key]["detail_in_evidence_record"] = True
                continue
            if summary and key in {"cases", "case_inputs", "measurement"}:
                result[key + "_detail"] = {"total": len(item), "retrieval": "calculate with calculation_offset and calculation_limit"}
                continue
            if isinstance(item, (list, tuple)):
                result[key] = [compact_calculation(row, limit=limit, summary=summary) for row in item[offset:offset + limit]]
                if len(item) > limit or offset:
                    result[key + "_page"] = {"offset": offset, "total": len(item),
                        "next_offset": offset + limit if offset + limit < len(item) else None}
            elif isinstance(item, dict) and key in {"cases", "alternatives", "facts", "observations", "witnesses"}:
                entries = list(item.items())
                result[key] = {k: compact_calculation(v, limit=limit, summary=summary) for k, v in entries[offset:offset + limit]}
                if len(entries) > limit or offset:
                    result[key + "_page"] = {"offset": offset, "total": len(entries),
                        "next_offset": offset + limit if offset + limit < len(entries) else None}
            else:
                result[key] = compact_calculation(item, limit=limit, summary=summary)
        return result
    if isinstance(value, (list, tuple)):
        return [compact_calculation(row, limit=limit, summary=summary) for row in value[:limit]]
    return value
