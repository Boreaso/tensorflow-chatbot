import math
import time

from . import misc_utils as utils


def init_stats():
    """Initialize statistics that we want to keep."""
    return {"step_time": 0.0, "loss": 0.0, "predict_count": 0.0,
            "total_count": 0.0, "grad_norm": 0.0, "best_bleu": 0.0}


def update_stats(stats, summary_writer, start_time, step_result, best_bleu):
    """Update stats: write summary and accumulate statistics."""
    (step_loss, step_predict_count, step_summary, global_step,
     step_word_count, batch_size, grad_norm, learning_rate) = step_result

    # Write step summary.
    summary_writer.add_summary(step_summary, global_step)

    # update statistics
    stats["step_time"] += (time.time() - start_time)
    stats["loss"] += (step_loss * batch_size)
    stats["predict_count"] += step_predict_count
    stats["total_count"] += float(step_word_count)
    stats["grad_norm"] += grad_norm
    stats["learning_rate"] = learning_rate
    stats["best_bleu"] = best_bleu

    return global_step


def check_stats(stats, global_step, steps_per_stats):
    """Print statistics and also check for overflow."""
    # Print statistics for the previous epoch.
    avg_step_time = stats["step_time"] / steps_per_stats
    avg_grad_norm = stats["grad_norm"] / steps_per_stats
    train_ppl = utils.safe_exp(
        stats["loss"] / stats["predict_count"])
    speed = stats["total_count"] / (1000 * stats["step_time"])
    best_bleu = stats['best_bleu']
    print("  global step %d lr %g "
          "step-time %.2fs wps %.2fK ppl %.2f gN %.2f bleu %d" %
          (global_step, stats["learning_rate"],
           avg_step_time, speed, train_ppl, avg_grad_norm, best_bleu))

    # Check for overflow
    is_overflow = False
    if math.isnan(train_ppl) or math.isinf(train_ppl) or train_ppl > 1e20:
        print("  step %d overflow, stop early" % global_step)
        is_overflow = True

    return is_overflow
