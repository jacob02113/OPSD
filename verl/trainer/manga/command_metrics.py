"""Raw-proposal diagnostics and rates computed from aggregated counts."""

PROPOSAL_KINDS = ("enter", "detect", "read", "link", "ground", "invalid")
PROPOSAL_METRICS = (
    "proposal_sampled", "action_illegal", "action_legal", "syntax_illegal",
    "object_checked", "object_illegal",
)
# Preserve the historical 5 * 4 command counters and enter_dependency_blocked.
PROPOSAL_STATS_OFFSET = 21


def proposal_stats(counters):
    return [counters[f"proposal_{kind}_{name}"]
            for kind in PROPOSAL_KINDS for name in PROPOSAL_METRICS]


def add_proposal_rates(metrics):
    """Called by the driver after microbatch/rank/trigger aggregation.

    Object legality is conditional on an allowed action and valid syntax.
    Zero-denominator rates are zero; their checked counts are logged as well.
    """
    prefix = "actor/manga_command/"
    if prefix + "enter_proposal_sampled" not in metrics:
        return
    totals = dict.fromkeys(PROPOSAL_METRICS, 0.0)
    for kind in PROPOSAL_KINDS:
        values = {name: float(metrics[prefix + kind + "_" + name])
                  for name in PROPOSAL_METRICS}
        for name, value in values.items():
            totals[name] += value
        for name, denominator in (("action_illegal", "proposal_sampled"),
                                  ("syntax_illegal", "action_legal"),
                                  ("object_illegal", "object_checked")):
            metrics[prefix + kind + "_" + name + "_rate"] = (
                values[name] / values[denominator] if values[denominator] else 0.0)
        if prefix + kind + "_sampled" in metrics:
            sampled = float(metrics[prefix + kind + "_sampled"])
            metrics[prefix + kind + "_illegal_rate"] = (
                float(metrics[prefix + kind + "_illegal"]) / sampled if sampled else 0.0)
    for name, value in totals.items():
        metrics["actor/manga_intent/" + name] = value
    for name, denominator in (("action_illegal", "proposal_sampled"),
                              ("syntax_illegal", "action_legal"),
                              ("object_illegal", "object_checked")):
        metrics["actor/manga_intent/" + name + "_rate"] = (
            totals[name] / totals[denominator] if totals[denominator] else 0.0)

