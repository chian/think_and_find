"""Scientific-object lookup and reproducible formal tool calls.

These are tools selected by Scientific and its three scientific children. They
do not contain agent loops, choose research tasks, or award scientific support.
Lean verifies the equivalence statement constructed by this adapter. Its proof
and source-to-formal bindings remain inspectable in an immutable host receipt.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import uuid

from jsonschema import Draft202012Validator, ValidationError

from .importance import evaluate_cases
from .records import ResearchResult, identity


RECEIPT_FIELDS = ("tool", "question", "inputs", "request", "status", "result", "artifacts", "limitations")
TOOLS = ("lookup", "lean_equivalence", "causal_calculation", "compare_tasks")
STATUSES = ("found", "verified_equivalent", "computed", "partial", "unresolved", "unavailable", "error")
_TEXT = {"type": "string", "minLength": 1}
_REF = {"type": "object", "additionalProperties": False,
        "required": ["key", "version"],
        "properties": {"key": _TEXT, "version": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}}
_SOURCE_BINDING = {"source": _REF, "selector": {"type": "array", "minItems": 1, "items": _TEXT}}
_SCIENTIFIC_KINDS = {
    "inference", "premise", "theory", "application", "theory_fragment", "reasoning", "comparison", "repair",
    "expectation", "expectation_break", "expectation_support", "support_contribution", "causal_change",
    "importance", "importance_result", "source_formalization", "expectation_test", "support_assessment",
    "repair_assessment", "examination", "discovery_goal", "discovery_finding", "discovery_check", "tool_receipt", "task_result",
}
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_FORBIDDEN_LEAN = re.compile(
    r"\b(?:sorry\w*|admit\w*|axiom\w*|unsafe\w*|native_decide|ofReduceBool|ofReduceNat|trustCompiler|"
    r"run_tac|run_elab|elab|macro|syntax|initialize|builtin_initialize|import|namespace|section|end|"
    r"theorem|lemma|def|abbrev|opaque|instance|structure|class|attribute|set_option|open|export|"
    r"command_elab|implemented_by|extern|evalExpr|evalTerm|compileDecl)\b|[#`\";]"
)
_PROOF_HEADS = {
    "rfl", "simp", "simp_all", "simpa", "ring", "ring_nf", "field_simp", "norm_num", "omega", "aesop",
    "linarith", "nlinarith", "constructor", "intro", "intros", "apply", "exact", "refine", "rw", "rwa",
    "change", "unfold", "dsimp", "cases", "rcases", "obtain", "have", "show", "first", "all_goals",
    "any_goals", "case", "next", "focus", "congr", "funext", "ext", "decide", "contradiction", "assumption",
    "exfalso", "classical", "left", "right", "trivial", "split", "subst", "rename_i", "rename",
}
_STANDARD_AXIOMS = {"propext", "Quot.sound", "Classical.choice"}


TOOL_INSTRUCTIONS = """Scientific tools are selected for a specific scientific question, not mandatory stages.
Use action=scientific_tool with name=lookup to select stored scientific records, Gs, expected
generalizability bounds, comparisons or repairs. Supply exact references or scientific record
kinds/query; this is not literature search or source-text retrieval. Returned current records
become selected scientific inputs and historical versions become selected snapshots.
Use name=lean_equivalence to check whether two formal expressions are equal, or propositions
are logically equivalent, under explicit source-linked assumptions. Supply Lean parameters
bound to scientific record fields, left/right terms, relation=equal or iff, and a proposed
tactic proof. The adapter constructs the theorem itself and the real Lean kernel checks it.
The result is verified_equivalent, unresolved, unavailable or error; a failed proof never
establishes inequivalence. Lean uses the configured Std/Mathlib library; submit ordinary proof
tactics, not declarations, axioms, imports, sorry/admit or compiler-evaluation shortcuts.
Use name=causal_calculation with an admitted importance protocol key, declared case_ids and
an optional evaluation_context to execute its SCM, equations or discrete simulation. The
protocol specifies the original and altered relationship and the outcomes being compared.
Lookup and each formal call return an exact-input tool_receipt. Inspect the result and use it
to revise the reconstruction/repair/comparison, retain an established formal result, or report
the missing premise. A tool receipt alone does not establish source fidelity or empirical truth,
and invoking a tool does not itself earn support/change credit. The current SCM evaluator covers
deterministic acyclic structural models; unsupported feedback/dynamics remain explicit errors.
Use name=compare_tasks with a causal_change record and its declared case_ids to execute
the original, candidate and reference predictions for one task. The record fixes exact G
and original expected generalizability bounds, source-bound models, task quantity/units,
accuracy tolerances and circumstances. It supports explicit mathematical equations and
ODE directional-crossing times with SciPy DOP853 and tolerance refinement. Declare ODE
initial states, event expression, direction and the physical experiment's time interval;
this model interval is not an episode deadline. A crossing beyond the reduced initial state
is explicitly not_represented; no crossing in the experiment interval is unresolved, never
proof of absence. The host computes applicability-difference and gradual improvement credit.
Original-G support, a full repair, a relation reversal or failed competing alternatives are
not prerequisites. Retain losses on other tasks and all original failures. Numerical refinement
estimates are not rigorous error bounds or empirical validation. For analytical claims without
an executed comparison, retain the public derivation as unscored scientific reasoning.
"""


def _object(properties):
    return {"type": "object", "additionalProperties": False,
            "required": list(properties), "properties": properties}


def tool_schema():
    """Closed primitive-operation branches; every scientific episode receives these."""
    arguments = {
        "lookup": _object({"keys": {"type": "array", "items": _REF},
                           "kinds": {"type": "array", "uniqueItems": True, "items": _TEXT},
                           "query": {"type": "string"}}),
        "lean_equivalence": _object({
            "relation": {"enum": ["equal", "iff"]},
            "parameters": {"type": "array", "items": _object({
                "name": _TEXT, "type": _TEXT, **_SOURCE_BINDING})},
            "assumptions": {"type": "array", "items": _object({
                "name": _TEXT, "proposition": _TEXT, **_SOURCE_BINDING})},
            "left": _TEXT, "right": _TEXT, "proof": _TEXT}),
        "causal_calculation": _object({
            "assessment": _TEXT, "case_ids": {"type": "array", "minItems": 1, "uniqueItems": True,
                                               "items": _TEXT},
            "evaluation_context": {"type": "string"}}),
        "compare_tasks": _object({"assessment": _TEXT,
            "case_ids": {"type": "array", "minItems": 1, "uniqueItems": True, "items": _TEXT}}),
    }
    return {"oneOf": [_object({"action": {"const": "scientific_tool"}, "name": {"const": name},
        "question": _TEXT, "inputs": {"type": "array", "items": _REF},
        "arguments": schema, "rationale": _TEXT}) for name, schema in arguments.items()]}


def receipt_references(data):
    return sorted({ref["key"] for ref in data["inputs"]})


def receipt_snapshot_references(data):
    return deepcopy(data["inputs"])


def receipt_context_pins(data):
    pins = {}
    for ref in data["inputs"]:
        pins.setdefault(ref["key"], set()).add(ref["version"])
    return pins


def validate_receipt(sheet, data):
    if not isinstance(data, dict) or set(data) != set(RECEIPT_FIELDS):
        raise ValueError("A scientific tool receipt requires its exact host-generated fields")
    if data["tool"] not in TOOLS or data["status"] not in STATUSES:
        raise ValueError("Unknown scientific tool/result status")
    if not isinstance(data["question"], str) or not data["question"].strip():
        raise ValueError("Scientific tool receipts retain the actual question")
    if not isinstance(data["inputs"], list):
        raise ValueError("Scientific tool inputs are exact record snapshots")
    for ref in data["inputs"]:
        if not isinstance(ref, dict) or set(ref) != {"key", "version"}:
            raise ValueError("Scientific tool inputs require exact key/version references")
        row = sheet.snapshot(ref)
        if row["kind"] not in _SCIENTIFIC_KINDS:
            raise ValueError("Scientific tools use scientific records, not raw source evidence")
    if any(not isinstance(data[name], dict) for name in ("request", "result", "artifacts")):
        raise ValueError("Tool requests, results and artifacts are explicit objects")
    if not isinstance(data["limitations"], list) or any(not isinstance(item, str) for item in data["limitations"]):
        raise ValueError("Tool limitations are explicit statements")
    try:
        Draft202012Validator(tool_schema()).validate({"action": "scientific_tool", "name": data["tool"],
            "question": data["question"], "inputs": data["inputs"], "arguments": data["request"],
            "rationale": "Retained host execution receipt"})
    except ValidationError as exc:
        raise ValueError("A scientific tool receipt must retain its typed request") from exc
    result = data["result"]
    if data["status"] == "verified_equivalent":
        if data["tool"] != "lean_equivalence" or result.get("returncode") != 0:
            raise ValueError("Verified equivalence requires a successful real Lean execution")
        axioms = result.get("axioms")
        if not isinstance(axioms, list) or not set(axioms) <= _STANDARD_AXIOMS:
            raise ValueError("Lean equivalence receipts must retain only permitted standard axioms")
        report = result.get("axiom_report", "")
        reported = re.search(r"'scientific_equivalence' depends on axioms:\s*\[([^\]]*)\]", report)
        no_axioms = "'scientific_equivalence' does not depend on any axioms" in report
        report_axioms = {value.strip() for value in reported.group(1).split(",") if value.strip()} if reported else set()
        if not (reported or no_axioms) or report_axioms != set(axioms):
            raise ValueError("Lean equivalence requires the actual theorem's retained axiom report")
        code, formal = _lean_source(sheet, {"inputs": data["inputs"], "arguments": data["request"]},
                                    result["backend"]["imports"])
        if (hashlib.sha256(code.encode()).hexdigest() != data["artifacts"].get("lean_source_sha256")
                or formal["proposition"] != result.get("proposition") or formal["bindings"] != result.get("bindings")):
            raise ValueError("Lean receipt does not match its exact scientific bindings and generated equivalence")
    if data["tool"] == "causal_calculation" and data["status"] in {"computed", "partial", "unresolved"}:
        for case in result["cases"]:
            row = sheet.snapshot({"key": case["key"], "version": case["version"]})
            if row["kind"] != "importance_result" or row["data"] != {
                    key: value for key, value in case.items() if key not in {"key", "version"}}:
                raise ValueError("Causal tool results must match their retained execution receipts")
    if data["tool"] == "compare_tasks" and data["status"] in {"computed", "partial", "unresolved"}:
        for ref in result["cases"]:
            row = sheet.snapshot(ref)
            if row["kind"] != "task_result" or row["data"]["assessment"] != data["request"]["assessment"]:
                raise ValueError("Task-comparison receipts require their actual declared executed results")


def _run_directory(source):
    resource = os.environ.get("THINK_AND_FIND_WORKFLOW_RESOURCES")
    if resource:
        path = Path(json.loads(Path(resource).read_text())["run_directory"]).resolve()
    else:
        checkpoint = getattr(source.run.providers, "checkpoint", None)
        if checkpoint is None:
            raise ValueError("Scientific tool artifacts require a launcher-owned run directory")
        path = Path(checkpoint.path).resolve().parent
    root = Path(__file__).resolve().parents[2]
    if not path.is_relative_to(root / "runs"):
        raise ValueError("Scientific tool artifacts belong beneath this repository's runs/")
    return path


def _directory(source):
    directory = _run_directory(source) / "scientific-tools" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    return directory


def _authorize_inputs(source, refs, *, lookup=False):
    sheet = source.ledger.causal_sheet
    for ref in refs:
        row = sheet.snapshot(ref)
        if row["kind"] not in _SCIENTIFIC_KINDS:
            raise ValueError("Scientific tool inputs select scientific objects, not source text")
        # Formal calls use already selected/commissioned objects. Lookup itself
        # is the explicit operation that can expose an additional stored object.
        if not lookup:
            scoped = {(item["key"], item["version"]) for item in source.scoped_snapshots()}
            keys = set(source.selected_keys) | set(getattr(source.commission, "inputs", ()))
            if ref["key"] not in keys and (ref["key"], ref["version"]) not in scoped:
                raise ValueError(f"Lookup or select scientific input {ref['key']!r} before using it")


def _lookup(source, request):
    sheet, arguments = source.ledger.causal_sheet, request["arguments"]
    kinds = set(arguments["kinds"])
    if kinds - _SCIENTIFIC_KINDS:
        raise ValueError("Lookup record kinds must be scientific objects, not raw source text")
    refs = {(ref["key"], ref["version"]): dict(ref) for ref in arguments["keys"]}
    query = arguments["query"].casefold().strip()
    for key, row in sheet.rows.items():
        if row["kind"] not in _SCIENTIFIC_KINDS or (kinds and row["kind"] not in kinds):
            continue
        searchable = json.dumps({"key": key, "data": row["data"]}, ensure_ascii=False).casefold()
        if (kinds or query) and (not query or query in searchable):
            refs[key, row["version"]] = {"key": key, "version": row["version"]}
    matches = []
    historical = {(ref["key"], ref["version"]): ref for ref in source.selected_snapshots}
    for ref in refs.values():
        row = sheet.snapshot(ref)
        if row["kind"] not in _SCIENTIFIC_KINDS:
            raise ValueError("Lookup does not return source-text records")
        matches.append({**ref, "kind": row["kind"]})
        if row["version"] == sheet.rows[ref["key"]]["version"]:
            if ref["key"] not in source.selected_keys:
                source.selected_keys.append(ref["key"])
        else:
            historical[ref["key"], ref["version"]] = dict(ref)
    source.selected_snapshots = list(historical.values())
    return ("found" if matches else "unresolved"), {"matches": matches,
        "delivery": "Selected scientific objects are supplied in the next selected-sheet/historical input."}, {}, []


def _source_field(sheet, item, selected):
    ref = item["source"]
    if (ref["key"], ref["version"]) not in selected:
        raise ValueError("Every Lean parameter/assumption binding must name an exact selected input")
    value = sheet.snapshot(ref)["data"]
    for part in item["selector"]:
        value = value[int(part)] if isinstance(value, list) and part.isdigit() else value[part]
    return value


def _lean_fragment(text):
    if _FORBIDDEN_LEAN.search(text) or "/*" in text or "/-" in text or "--" in text:
        raise ValueError("Lean equivalence accepts terms/tactics, not declarations, comments, axioms or unchecked shortcuts")
    return text.strip()


def _lean_source(sheet, request, imports):
    arguments = request["arguments"]
    selected = {(ref["key"], ref["version"]) for ref in request["inputs"]}
    if not selected:
        raise ValueError("A Lean equivalence must identify the scientific records being compared")
    names, binders, mapped = set(), [], []
    for item in arguments["parameters"]:
        name = item["name"]
        if not _IDENTIFIER.fullmatch(name) or name in names:
            raise ValueError("Lean parameter names must be distinct explicit identifiers")
        type_text = _lean_fragment(item["type"])
        tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[ℝℚℕℤ]", type_text)
        if any(token not in {"Nat", "Int", "Rat", "Bool", "Prop", "Real", "ℝ", "ℚ", "ℕ", "ℤ"}
               for token in tokens) or re.search(r"[^A-Za-z0-9_ℝℚℕℤ ()→×]", type_text):
            raise ValueError("Use explicit scalar/function Lean types over Nat, Int, Rat, Bool, Prop or Real")
        mapped.append({**deepcopy(item), "scientific_value": _source_field(sheet, item, selected)})
        names.add(name)
        binders.append(f"({name} : {type_text})")
    for item in arguments["assumptions"]:
        name = item["name"]
        if not _IDENTIFIER.fullmatch(name) or name in names:
            raise ValueError("Lean assumption names must be distinct explicit identifiers")
        proposition = _lean_fragment(item["proposition"])
        mapped.append({**deepcopy(item), "scientific_value": _source_field(sheet, item, selected)})
        names.add(name)
        binders.append(f"({name} : {proposition})")
    left, right = (_lean_fragment(arguments[name]) for name in ("left", "right"))
    relation = "=" if arguments["relation"] == "equal" else "↔"
    proposition = f"({left}) {relation} ({right})"
    normalized = re.sub(r"\s+|[()]", "", proposition)
    for assumption in arguments["assumptions"]:
        if re.sub(r"\s+|[()]", "", assumption["proposition"]) == normalized:
            raise ValueError("The requested equivalence cannot itself be a supplied assumption")
    proof = _lean_fragment(arguments["proof"])
    for line in proof.splitlines():
        line = line.strip()
        if not line:
            continue
        head = re.match(r"[A-Za-z_][A-Za-z0-9_]*", line)
        if head is None or head.group() not in _PROOF_HEADS:
            raise ValueError("Use ordinary supported Lean tactic lines; arbitrary metaprograms/declarations are not accepted")
    imports_text = "\n".join("import " + name for name in imports)
    proof_text = "\n".join("  " + line for line in proof.splitlines())
    code = (imports_text + "\nset_option autoImplicit false\n\n"
            + "theorem scientific_equivalence " + " ".join(binders) + " :\n  "
            + proposition + " := by\n" + proof_text + "\n\n#print axioms scientific_equivalence\n")
    return code, {"proposition": proposition, "bindings": mapped,
                  "theorem": "scientific_equivalence"}


def _lean_config():
    root = Path(__file__).resolve().parents[2]
    file = Path(os.environ.get("THINK_AND_FIND_LEAN_CONFIG", root / "runs/scientific_tools/lean.json"))
    if file.is_file():
        config = json.loads(file.read_text())
    else:
        binary = shutil.which("lean")
        if not binary:
            return None
        config = {"executable": binary, "imports": ["Std"], "lean_path": []}
    if set(config) - {"executable", "imports", "lean_path", "version", "toolchain_origin"}:
        raise ValueError("Unknown Lean backend configuration fields")
    binary = Path(config["executable"]).resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        return None
    imports = config.get("imports", ["Std"])
    if not isinstance(imports, list) or not imports or any(
            not isinstance(name, str) or not re.fullmatch(r"(?:Std|Mathlib)(?:\.[A-Za-z0-9_]+)*", name)
            for name in imports):
        raise ValueError("Lean imports must be configured Std/Mathlib modules")
    paths = config.get("lean_path", [])
    if not isinstance(paths, list) or any(not isinstance(path, str) for path in paths):
        raise ValueError("Lean library paths must be explicit paths")
    return {**config, "executable": str(binary), "imports": imports, "lean_path": paths}


def _lean(source, request, directory):
    config = _lean_config()
    imports = config["imports"] if config else ["Std"]
    code, formal = _lean_source(source.ledger.causal_sheet, request, imports)
    lean_file = directory / "Equivalence.lean"
    lean_file.write_text(code)
    artifacts = {"lean_source": str(lean_file), "lean_source_sha256": hashlib.sha256(code.encode()).hexdigest()}
    limitations = ["Kernel checking verifies the encoded equivalence under the listed assumptions; the source-to-formal bindings are explicit scientific interpretations."]
    if config is None:
        return "unavailable", {**formal, "error": "Lean executable is unavailable; configure runs/scientific_tools/lean.json or THINK_AND_FIND_LEAN_CONFIG."}, artifacts, limitations
    env = dict(os.environ)
    if config["lean_path"]:
        env["LEAN_PATH"] = os.pathsep.join(config["lean_path"])
    with (directory / "lean.stdout").open("w") as stdout, (directory / "lean.stderr").open("w") as stderr:
        completed = subprocess.run([config["executable"], str(lean_file)], cwd=directory,
                                   env=env, stdout=stdout, stderr=stderr, check=False)
    artifacts.update(stdout=str(directory / "lean.stdout"), stderr=str(directory / "lean.stderr"))
    output = (directory / "lean.stdout").read_text() + (directory / "lean.stderr").read_text()
    axiom_match = re.search(r"'scientific_equivalence' depends on axioms:\s*\[([^\]]*)\]", output)
    no_axioms = "'scientific_equivalence' does not depend on any axioms" in output
    axioms = {value.strip() for value in axiom_match.group(1).split(",") if value.strip()} if axiom_match else set()
    checked = completed.returncode == 0 and (no_axioms or axiom_match is not None) and axioms <= _STANDARD_AXIOMS
    result = {**formal, "returncode": completed.returncode, "axioms": sorted(axioms), "axiom_report": output,
              "backend": {key: value for key, value in config.items() if key != "lean_path"}}
    if not checked:
        result["error"] = output or "Lean did not produce a checked equivalence and standard-axiom report"
    return ("verified_equivalent" if checked else "unresolved"), result, artifacts, limitations


def _causal(source, request):
    arguments, sheet = request["arguments"], source.ledger.causal_sheet
    assessment = arguments["assessment"]
    if not any(ref["key"] == assessment and ref["version"] == sheet.rows[assessment]["version"]
               for ref in request["inputs"]):
        raise ValueError("The causal calculation must identify its exact admitted protocol in inputs")
    source.authorize_evaluation(assessment)
    staged = type(sheet)(sheet.history)
    updates = []
    for case_id in arguments["case_ids"]:
        calculated = evaluate_cases(staged, assessment, [case_id], context=arguments["evaluation_context"] or None)
        staged.merge(calculated)
        updates.extend(calculated)
    result = {"assessment": {"key": assessment, "version": sheet.rows[assessment]["version"]},
              "method": sheet.rows[assessment]["data"]["method"],
              "cases": [{"key": row["key"], "version": row["version"], **row["data"]} for row in updates]}
    errors = [row["data"]["error"] for row in updates if row["data"]["error"]]
    status = "computed" if not errors else "unresolved" if len(errors) == len(updates) else "partial"
    return updates, status, result, {}, [
        "Executed consequences are conditional on the declared model and circumstances, not proof of empirical source fidelity.",
        "SCM evaluation is deterministic and acyclic; simulation evaluation uses the explicitly declared discrete model."]


def _record(source, data, updates=()):
    sheet = type(source.ledger.causal_sheet)(source.ledger.causal_sheet.history)
    for row in updates:
        sheet.merge((row,))
    semantic = {key: data[key] for key in RECEIPT_FIELDS if key not in {"artifacts", "question"}}
    key = "tool-receipt:" + identity(semantic)[:32]
    if key in sheet.rows:
        row = sheet.rows[key]
    else:
        row = sheet.put(key, "tool_receipt", data, dependencies=receipt_references(data))
    # Tool receipts are scientific-sheet records, not the separate physical
    # metric calculation records consumed by Ledger.calculations. Select the
    # receipt so the invoking episode receives its result in the next decision.
    if key not in source.selected_keys:
        source.selected_keys.append(key)
    return ResearchResult(evaluated=data["status"] not in {"error", "unavailable"},
        causal_sheet_updates=tuple([*updates, row]))


def _compare_tasks(source, request):
    from .task_comparisons import evaluate_cases as compare_cases, grade
    sheet = type(source.ledger.causal_sheet)(source.ledger.causal_sheet.history)
    key = request["arguments"]["assessment"]
    if sheet.anchor(key) not in request["inputs"]:
        raise ValueError("Select the exact causal_change record in compare_tasks.inputs")
    updates = compare_cases(sheet, key, request["arguments"]["case_ids"])
    unresolved = any(row["data"][role]["status"] == "unresolved"
                     for row in updates for role in ("original", "candidate", "reference"))
    return updates, "partial" if unresolved else "computed", {
        "assessment": key, "cases": [sheet.anchor(row["key"]) for row in updates],
        "comparison": grade(sheet, key)}, {}, [
        "Executed model consequences and tolerance-refinement estimates; not empirical validation or rigorous ODE error enclosures."]


def execute_tool(source, proposal):
    """Execute one selected tool and return its original iterator outcome."""
    try:
        Draft202012Validator(tool_schema()).validate(proposal)
    except ValidationError as exc:
        raise ValueError(f"Scientific tool request violates its typed contract at {list(exc.absolute_path)}: {exc.message}") from exc
    request = deepcopy(proposal)
    directory = _directory(source)
    (directory / "request.json").write_text(json.dumps(request, indent=2, ensure_ascii=False) + "\n")
    updates = ()
    try:
        _authorize_inputs(source, request["inputs"], lookup=request["name"] == "lookup")
        if request["name"] == "lookup":
            status, result, artifacts, limitations = _lookup(source, request)
        elif request["name"] == "lean_equivalence":
            status, result, artifacts, limitations = _lean(source, request, directory)
        elif request["name"] == "compare_tasks":
            updates, status, result, artifacts, limitations = _compare_tasks(source, request)
        else:
            updates, status, result, artifacts, limitations = _causal(source, request)
    except (ValueError, TypeError, KeyError, OSError, IndexError) as exc:
        status, result, artifacts, limitations = "error", {"error": str(exc)}, {}, []
    except ImportError as exc:
        status, result, artifacts, limitations = "unavailable", {"error": str(exc)}, {}, [
            "Install requirements-scientific.txt in the execution environment to use numerical ODE comparison."]
    data = {"tool": request["name"], "question": request["question"], "inputs": request["inputs"],
            "request": request["arguments"], "status": status, "result": result,
            "artifacts": {"request": str(directory / "request.json"), **artifacts}, "limitations": limitations}
    (directory / "result.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return _record(source, data, updates)
