"""Source-local ranking fixtures, not scientific evidence."""

import unittest

from episodic_workflow.case_reports.research.documents import Span, rank_spans


def spans(*texts):
    return tuple(Span(f"chunk:{index}", index * 100, (index + 1) * 100, text)
                 for index, text in enumerate(texts))


class LexicalRankingTests(unittest.TestCase):
    def test_symbols_and_numbers_do_not_match_inside_words(self):
        chunks = spans("exposure 2017 " * 50, *(["unrelated"] * 10), "x 7")
        self.assertEqual(rank_spans(chunks, "x 7")[0], chunks[-1])

    def test_rare_terms_outweigh_repeated_common_words(self):
        chunks = spans("the room " * 100, *(["the room discussion"] * 10),
                       "Beyond the well mixed room")
        self.assertEqual(rank_spans(chunks, "Beyond the well mixed room")[0], chunks[-1])

    def test_neighboring_context_lifts_data_after_a_heading(self):
        chunks = spans(*(["unrelated background"] * 10), "entrainment coefficient",
                       "Measured values and their uncertainty are reported here.",
                       *(["unrelated discussion"] * 10))
        ranked = rank_spans(chunks, "entrainment coefficient")
        self.assertEqual(ranked[0], chunks[10])
        self.assertLess(ranked.index(chunks[11]), ranked.index(chunks[0]))

    def test_no_threshold_discards_unmatched_chunks(self):
        chunks = spans("first", "target", "last")
        self.assertCountEqual(rank_spans(chunks, "target"), chunks)
        for query in ("", "!!!", "absent"):
            with self.subTest(query=query):
                self.assertEqual(rank_spans(chunks, query), chunks)
        self.assertEqual(rank_spans((), "target"), ())

    def test_equal_evidence_has_stable_document_order(self):
        chunks = spans("same evidence", "same evidence")
        self.assertEqual(rank_spans(chunks, "evidence"), chunks)
        self.assertEqual(rank_spans(chunks[:1], "evidence"), chunks[:1])
