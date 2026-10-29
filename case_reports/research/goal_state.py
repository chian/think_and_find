"""Shared support ledger and proportional change credit for existing Episodes."""
from fractions import Fraction

from ...credit_assignment.causal_support import compile_support, controller_projection
from .records import Credit, Ledger, ResearchResult

GLOBAL_GOAL_PROGRESS = "global_goal_progress"


class ScientificGoalState:
    def __init__(self, initial=None):
        self.ledger = Ledger(initial or ResearchResult())

    def commit(self, result):
        self.ledger.merge(result)
        return self.preview()

    def credit(self, result):
        """Scientific parent's support and proportional-change stock."""
        score = compile_support(Ledger(result).causal_sheet)
        points = {key: Fraction(value) for key, value in score["points"].items()}
        projection = controller_projection(points, score["resolution"])
        if not projection["supported"]:
            return Credit(evaluated=False)
        return Credit(evaluated=result.evaluated,
            quality_resolution=projection["resolution"],
            qualities=tuple((GLOBAL_GOAL_PROGRESS, key, value) for key, value in sorted(projection["values"].items())))

    def preview(self):
        calculation = self.ledger.causal_sheet.calculate()
        score = calculation["support_credit"]
        projection = controller_projection(score["points"], score["resolution"])
        return {"scientific_calculations": calculation,
                "investigation_score": float(Fraction(score["total"])),
                "root_progress_binding": score["policy"] if projection["supported"] else "unsupported_controller_precision",
                "controller_limitation": None if projection["supported"] else projection["reason"],
                "meaning": score["meaning"]}
