"""
conclude_metrics.py

Scans all session directories in AB_test_results, parses every
summary_comparison_all.txt, and produces a single aggregated report
across all sessions.
"""

import os
import re
import numpy as np
from pathlib import Path
from datetime import datetime
from A_B_utils import send_slack_message
from config import SLACK_CHANNEL_ID_2
from datetime import time
try:
    from config import IST
except ImportError:
    from datetime import timezone, timedelta
    IST = timezone(timedelta(hours=5, minutes=30))

BASE_PATH = "/home/ubuntu/workspace/A_B"
RESULTS_DIR = os.path.join(BASE_PATH, "AB_test_results")


def parse_summary_file(filepath):   
    """
    Parse a summary_comparison_all.txt and return a list of per-profile dicts.
    Each dict: {pid, nonubp_mean, ubp_mean, nonubp_p75, ubp_p75}
    """
    profiles = []
    with open(filepath, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        # Stop at aggregate section
        if line.startswith("="):
            break
        # Skip header, separator, empty lines
        if not line or line.startswith("-"):
            continue
        if "PID" in line and "NonUBP" in line:
            continue  # header row

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 7:
            continue

        try:
            pid = parts[0]
            nonubp_mean = float(parts[1])
            ubp_mean = float(parts[2])
            # parts[3] is Improvement (skip)
            nonubp_p75 = float(parts[4])
            ubp_p75 = float(parts[5])
            # parts[6] is Improvement (skip)

            profiles.append({
                "pid": pid,
                "nonubp_mean": nonubp_mean,
                "ubp_mean": ubp_mean,
                "nonubp_p75": nonubp_p75,
                "ubp_p75": ubp_p75,
            })
        except (ValueError, IndexError):
            continue

    return profiles


def format_improvement(old_val, new_val):
    """Format improvement as +/-delta (+/-pct%)"""
    diff = old_val - new_val
    if old_val == 0:
        pct = 0.0
    else:
        pct = (diff / old_val) * 100
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.2f} ({sign}{pct:.2f}%)"


def conclude(test=True, full=True):
    """
    Aggregate all summary_comparison_all.txt reports and send to Slack.

    Args:
        test: If True, print to terminal instead of sending to Slack.
        full: If True, send the full report (all sections).
              If False, send only Header + GRAND AGGREGATE + OVERALL STATS.
    """
    results_path = Path(RESULTS_DIR)
    if not results_path.exists():
        print(f"❌ Results directory not found: {RESULTS_DIR}")
        return

    # Find all session dirs that have a summary file
    session_dirs = sorted([
        d for d in results_path.iterdir()
        if d.is_dir() and (d / "summary_comparison_all.txt").exists()
    ])

    if not session_dirs:
        print("❌ No summary files found in any session directory.")
        return

    all_profiles = []
    session_summaries = []

    for session_dir in session_dirs:
        summary_file = session_dir / "summary_comparison_all.txt"
        profiles = parse_summary_file(str(summary_file))

        if profiles:
            improved = [p for p in profiles if p["nonubp_p75"] > p["ubp_p75"]]
            degraded = [p for p in profiles if p["nonubp_p75"] <= p["ubp_p75"]]

            session_summaries.append({
                "session": session_dir.name,
                "total": len(profiles),
                "improved": len(improved),
                "degraded": len(degraded),
                "profiles": profiles,
            })
            all_profiles.extend(profiles)

    # ======================================================================
    # Compute aggregates (needed for both full and condensed)
    # ======================================================================
    improved_profiles = [p for p in all_profiles if p["nonubp_p75"] > p["ubp_p75"]]
    degraded_profiles = [p for p in all_profiles if p["nonubp_p75"] <= p["ubp_p75"]]

    total = len(all_profiles)
    n_improved = len(improved_profiles)
    n_degraded = len(degraded_profiles)
    pct_improved = (n_improved / total * 100) if total > 0 else 0

    overall_mean_nonubp = np.mean([p["nonubp_mean"] for p in all_profiles]) if all_profiles else 0
    overall_mean_ubp = np.mean([p["ubp_mean"] for p in all_profiles]) if all_profiles else 0
    overall_p75_nonubp = np.mean([p["nonubp_p75"] for p in all_profiles]) if all_profiles else 0
    overall_p75_ubp = np.mean([p["ubp_p75"] for p in all_profiles]) if all_profiles else 0

    # ======================================================================
    # Build Header section (always included)
    # ======================================================================
    header_lines = []
    header_lines.append("=" * 120)
    header_lines.append("  CONCLUDED METRICS — ALL SESSIONS AGGREGATED")
    header_lines.append("=" * 120)
    header_lines.append(f"  Generated at: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')} IST")
    header_lines.append(f"  Sessions analyzed: {len(session_summaries)}")
    header_lines.append(f"  Total profiles: {len(all_profiles)}")
    header_lines.append("")

    # ======================================================================
    # Build Per-Session Summary (full only)
    # ======================================================================
    session_lines = []
    session_lines.append("-" * 120)
    session_lines.append(f"  {'Session':<50} | {'Total':>6} | {'Improved':>8} | {'Degraded':>8} | {'Improved %':>10}")
    session_lines.append("-" * 120)
    for s in session_summaries:
        pct = (s["improved"] / s["total"] * 100) if s["total"] > 0 else 0
        session_lines.append(
            f"  {s['session']:<50} | {s['total']:>6} | {s['improved']:>8} | {s['degraded']:>8} | {pct:>9.1f}%"
        )
    session_lines.append("-" * 120)
    session_lines.append("")

    # ======================================================================
    # Build All Profiles Table (full only)
    # ======================================================================
    profiles_lines = []
    profiles_lines.append("=" * 120)
    profiles_lines.append("  ALL PROFILES")
    profiles_lines.append("=" * 120)
    prof_header = (
        f"  {'PID':<40} | {'NonUBP Mean dE':>14} | {'UBP Mean dE':>11} | {'Mean Improvement':>18} "
        f"| {'NonUBP P75 dE':>13} | {'UBP P75 dE':>10} | {'P75 Improvement':>18}"
    )
    profiles_lines.append(prof_header)
    profiles_lines.append("-" * 120)

    for p in all_profiles:
        mean_imp = format_improvement(p["nonubp_mean"], p["ubp_mean"])
        p75_imp = format_improvement(p["nonubp_p75"], p["ubp_p75"])
        profiles_lines.append(
            f"  {p['pid']:<40} | {p['nonubp_mean']:>14.2f} | {p['ubp_mean']:>11.2f} | {mean_imp:>18} "
            f"| {p['nonubp_p75']:>13.2f} | {p['ubp_p75']:>10.2f} | {p75_imp:>18}"
        )
    profiles_lines.append("-" * 120)
    profiles_lines.append("")

    # ======================================================================
    # Build Grand Aggregate (always included)
    # ======================================================================
    grand_lines = []
    grand_lines.append("=" * 120)
    grand_lines.append("  GRAND AGGREGATE SUMMARY (based on P75 Delta E)")
    grand_lines.append("=" * 120)
    agg_header = (
        f"  {'Group':<12} | {'Count':>6} | {'Mean NonUBP':>12} | {'Mean UBP':>12} | {'Mean Improvement':>20} "
        f"| {'P75 NonUBP':>12} | {'P75 UBP':>12} | {'P75 Improvement':>20}"
    )
    grand_lines.append(agg_header)
    grand_lines.append("-" * 120)

    for label, group in [("Improved", improved_profiles), ("Degraded", degraded_profiles)]:
        if group:
            mean_nonubp = np.mean([p["nonubp_mean"] for p in group])
            mean_ubp = np.mean([p["ubp_mean"] for p in group])
            p75_nonubp = np.percentile([p["nonubp_p75"] for p in group], 75)
            p75_ubp = np.percentile([p["ubp_p75"] for p in group], 75)
            mean_imp = format_improvement(mean_nonubp, mean_ubp)
            p75_imp = format_improvement(p75_nonubp, p75_ubp)
            grand_lines.append(
                f"  {label:<12} | {len(group):>6} | {mean_nonubp:>12.2f} | {mean_ubp:>12.2f} | {mean_imp:>20} "
                f"| {p75_nonubp:>12.2f} | {p75_ubp:>12.2f} | {p75_imp:>20}"
            )
        else:
            grand_lines.append(
                f"  {label:<12} | {0:>6} | {'N/A':>12} | {'N/A':>12} | {'N/A':>20} "
                f"| {'N/A':>12} | {'N/A':>12} | {'N/A':>20}"
            )

    grand_lines.append("-" * 120)
    grand_lines.append("")

    # ======================================================================
    # Build Overall Stats (always included)
    # ======================================================================
    overall_lines = []
    overall_lines.append("=" * 120)
    overall_lines.append("  OVERALL STATS")
    overall_lines.append("=" * 120)
    overall_lines.append(f"  Total Profiles:     {total}")
    overall_lines.append(f"  Improved:           {n_improved} ({pct_improved:.1f}%)")
    overall_lines.append(f"  Degraded:           {n_degraded} ({100 - pct_improved:.1f}%)")
    overall_lines.append(f"  Overall Mean dE:    NonUBP={overall_mean_nonubp:.2f}  UBP={overall_mean_ubp:.2f}  {format_improvement(overall_mean_nonubp, overall_mean_ubp)}")
    overall_lines.append(f"  Overall Mean P75:   NonUBP={overall_p75_nonubp:.2f}  UBP={overall_p75_ubp:.2f}  {format_improvement(overall_p75_nonubp, overall_p75_ubp)}")
    overall_lines.append("=" * 120)
    overall_lines.append("")

    # ======================================================================
    # Assemble report based on full flag
    # ======================================================================
    if full:
        all_lines = header_lines + session_lines + profiles_lines + grand_lines + overall_lines
    else:
        all_lines = header_lines + grand_lines + overall_lines

    report = "\n".join(all_lines)

    # Save full report to file always
    full_report = "\n".join(header_lines + session_lines + profiles_lines + grand_lines + overall_lines)
    output_path = os.path.join(RESULTS_DIR, "concluded_metrics_all.txt")
    with open(output_path, "w") as f:
        f.write(full_report)
    print(f"📄 Report saved to: {output_path}")

    # ======================================================================
    # Send to Slack or print
    # ======================================================================
    if test:
        print(report)
    else:
        # Send to Slack channel 2 in chunks
        MAX_CHARS = 3500
        lines_to_send = report.split('\n')
        chunks = []
        current_msg = ""

        for line in lines_to_send:
            line_len = len(line) + 1
            if len(current_msg) + line_len > MAX_CHARS:
                chunks.append(current_msg)
                current_msg = line + "\n"
            else:
                current_msg += line + "\n"
        if current_msg:
            chunks.append(current_msg)

        for i, chunk in enumerate(chunks):
            msg = f"```\n{chunk}\n```"
            if send_slack_message(msg, channel=SLACK_CHANNEL_ID_2):
                print(f"✅ Chunk {i+1}/{len(chunks)} sent to Slack")
            else:
                print(f"❌ Failed to send chunk {i+1}/{len(chunks)}")
            time.sleep(1)


if __name__ == "__main__":
    conclude(test=False, full=False)
