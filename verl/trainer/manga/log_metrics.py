"""Compact Manga training logs after rank and microbatch aggregation."""


def compact_manga_metrics(metrics):
    """Keep diagnostic counts; derive rates from totals, then drop redundant fields.

    This runs only at the logging boundary. Training tensors and worker metrics
    remain unchanged, and non-Manga runs are untouched.
    """
    intent = "actor/manga_intent/"
    if intent + "proposals" not in metrics:
        return

    def ratio(output, numerator, denominator):
        if numerator in metrics and denominator in metrics:
            count = float(metrics[denominator])
            metrics[output] = float(metrics[numerator]) / count if count else 0.0

    for name, numerator, denominator in (
        ("legal_rate", "legal_proposals", "proposals"),
        ("completion_rate", "completed_rollouts", "episodes"),
        ("gt_fallback_rate", "gt_fallbacks", "proposals"),
        ("retry_attempts_per_command", "retry_attempts", "proposals"),
        ("forced_tokens_per_command", "student_forced_tokens", "proposals"),
        ("retry_success_rate", "retry_successes", "retry_attempts"),
        ("tail_limit_rate", "teacher_tail_limit_rows", "teacher_scored_rows"),
    ):
        ratio(intent + name, intent + numerator, intent + denominator)
    # Retain denominators so zero rates can be distinguished from no observations.
    boundary = "actor/manga_boundary/"
    ratio(boundary + "early_eos_rate", boundary + "early_eos", boundary + "student_requests")
    ratio(boundary + "continue_error_rate", boundary + "continue_errors", boundary + "continue_boundaries")
    for kind in ("enter", "detect", "read", "link", "ground"):
        base = "actor/manga_command/" + kind
        ratio(base + "_illegal_rate", base + "_illegal", base + "_sampled")

    discard = {
        "actor/manga_opsd/loss",  # actor/loss includes the same boundary CE
        "actor/distillation/raw_kl_mean",  # manga_opsd/opsd_loss
        "actor/distillation/clipped_loss_mean",
        "actor/distillation/student_mass_max", "actor/distillation/teacher_mass_max",
        "perf/time_per_step",  # timing_s/step
    }
    discard.update(intent + name for name in (
        "exact_proposals", "legal_retained", "direct_target", "dag_prerequisite",
        "unmatched_target_fallback", "completed_target_fallback",
        "teacher_missing_mass_sum", "teacher_mixture_missing_mass_sum",
        "teacher_active_components_sum", "teacher_output_tail_mass_sum",
        "component_missing_mass", "teacher_mixture_missing_mass",
        "teacher_image_preprocess_reused", "teacher_submissions_during_preparation",
    ))
    discard.update(boundary + name for name in (
        "terminal_error_rate", "terminal_errors",  # command stop may hide EOS
        "mixed_boundary_tokens",
    ))
    discard.update("actor/manga_command/" + kind + "_changed"
                   for kind in ("enter", "detect", "read", "link", "ground"))
    for key in list(metrics):
        if key in discard or key.startswith((
            "actor/manga_tokens/", "global_seqlen/", "timing_per_token_ms/",
            "training/num_turns/",
        )):
            del metrics[key]
        elif key in ("timing_s/reward", "timing_s/adv") and float(metrics[key]) < 0.001:
            del metrics[key]

