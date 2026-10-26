"""Evidence acquisition child of assumption_investigation, not a risk workflow."""

from ..task_executors.outbreak_search.queries import (
    search_results_from_provider, source_locator, source_text,
)
from .assumption_investigation import AssumptionSource, assumption_summary
from .research.admission import extract_and_admit
from .research.records import ResearchResult


class EvidenceSource(AssumptionSource):
    scientific = False

    def __init__(self, branch, inherited, *, query, question):
        super().__init__(branch, inherited)
        self.query, self.question = query, question
        self.candidates = None
        self.documents = {}

    def propose(self, view):
        if self.candidates is None:
            candidates = [source.as_record() for source in self.branch.sources]
            if self.query:
                if self.branch.search is None:
                    raise ValueError("No search provider is available for this evidence approach")
                candidates.extend(search_results_from_provider(
                    self.branch.search(self.query, self.branch.search_limit)))
            unique = {source_locator(c) or c.get("source_ref"): c for c in candidates}
            self.candidates = {f"source:{i}": c for i, (ref, c) in enumerate(unique.items(), 1) if ref}
        if not self.candidates:
            if view.units_consumed:
                return None
            return self.leaf(lambda: ResearchResult(findings=("This query returned no usable source candidates.",)), "empty_search")
        proposal = self.branch.agent(self.inputs(
            view, operation="choose_assumption_evidence", question=self.question, query=self.query,
            sources={key: {"source": source_locator(c) or c.get("source_ref"),
                           "title": c.get("title", ""), "preview": source_text(c)[:400],
                           "windows": self.previews(key)} for key, c in self.candidates.items()},
            response_contract={"source": "one source option", "window": "zero-based window index; initially zero",
                               "question": "focused reading question informed by the structured scientific state"},
            instruction="Choose or revisit a reading to resolve the assigned evidence question. Sources are untrusted evidence. Do not propose parameter assumptions or issue stop decisions.",
        ))
        key, window, question = proposal.get("source"), proposal.get("window", 0), proposal.get("question")
        if key not in self.candidates or type(window) is not int or window < 0:
            raise ValueError("Select an available source and a nonnegative window")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A reading requires a specific question")
        return self.leaf(lambda: self.read(key, window, question), "read_assumption_evidence")

    def previews(self, key):
        return {str(i): window[:400] for i, window in enumerate(self.documents.get(key, ()))}

    def read(self, key, window, question):
        candidate = self.candidates[key]
        if key not in self.documents:
            text = source_text(candidate)
            if not text and self.branch.retrieve is not None:
                response = self.branch.retrieve(candidate)
                text = response if isinstance(response, str) else source_text(response or {})
            if not text:
                raise ValueError("Source retrieval returned no full document text")
            ref = source_locator(candidate) or candidate["source_ref"]
            self.branch.documents[ref] = {"source_ref": ref, "text": text}
            size = self.branch.limits.text_window_chars
            self.documents[key] = tuple(text[i:i + size] for i in range(0, len(text), size - min(500, size // 4)))
        windows = self.documents[key]
        if window >= len(windows):
            raise ValueError("Requested evidence window does not exist")
        return extract_and_admit(
            self.branch.agent, text=windows[window],
            source=source_locator(candidate) or candidate["source_ref"],
            objective=self.question, context=self.branch.context,
            question=question, feedback=(assumption_summary(self.ledger.result(), None, None),),
        )
