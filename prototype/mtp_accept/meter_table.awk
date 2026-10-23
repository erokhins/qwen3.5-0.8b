# meter_table.awk — renders one `--mtp-meter` run's phase lines and REFUSES the table when the run's own
# accounting does not hold. Used by `meter_round.sh`; exercised on synthetic logs by
# `meter_round_selftest.sh`, because a refusal that has never fired is decoration.
#
# Three arguments: -v role=<verify|accept> -v w=<lanes> -v logname=<path, for the message only>.
#
# The refusals, and the failure each one exists to catch:
#   * `meter=` on the RESULT line is not `ok`. Either a device wait failed (`sync_failed`) or the coverage
#     band was missed (`out_of_band`); in both cases the milliseconds bound nothing.
#   * coverage outside 0.8-1.2. `overhead` is DERIVED as wall - sum(phases) and is kept OUT of the sum, so
#     coverage is precisely "do the named phases account for the window" — a list that accounts for 40 % of
#     it is a list with a phase missing, and tabulating it would present a wrong attribution as a finding.
#   * zero phase lines for this role: the meter never armed, so a binary without the instrument would print
#     an empty table that reads as a null result.
#   * sum(phases) + overhead != wall (tolerance 2 µs/round). An arithmetic identity over the printed numbers:
#     it cannot pass if a phase is double-charged into the sum while `overhead` still carries it, or if a
#     phase lands inside the window but in no row.
#
# `not_metered` and `derived` rows are excluded from the sum by design: the first has no milliseconds at all,
# the second IS the residual. A sum that included `overhead` would make coverage 1.000 by construction, which
# is the failure shape `AGENTS.md` records for every gate that cannot fail.
BEGIN { FS = " "; nrows = 0 }
/^MTPMETER role=.* phase=instrument/ {
    for (i = 1; i <= NF; i++) { n = $i; sub(/=.*/, "", n); v = $i; sub(/^[^=]*=/, "", v); ins[n] = v }
}
/^RESULT mode=/ {
    for (i = 1; i <= NF; i++) { n = $i; sub(/=.*/, "", n); v = $i; sub(/^[^=]*=/, "", v); r[n] = v }
}
/^MTPMETER role=/ {
    ph = ""; pr = ""; runs = ""; spr = ""; sa = ""; st = ""; rl = ""
    for (i = 1; i <= NF; i++) {
        n = $i; sub(/=.*/, "", n); v = $i; sub(/^[^=]*=/, "", v)
        if      (n == "role")         rl = v
        else if (n == "phase")        ph = v
        else if (n == "ms_per_round") pr = v
        else if (n == "runs")         runs = v
        else if (n == "ms_per_run")   spr = v
        else if (n == "sync_added")   sa = v
        else if (n == "status")       st = v
        else if (n == "wall_ms_per_round") wpr = v
    }
    if (ph == "" || ph == "instrument" || rl != role) next
    nrows++
    name[nrows] = ph; prv[nrows] = pr; runsv[nrows] = runs; sprv[nrows] = spr; sav[nrows] = sa; stv[nrows] = st
    if (ph == "overhead") ov = pr + 0
    if (ph == "summary") { if (wpr != "") wall = wpr + 0 }
}
END {
    bad = 0
    if (r["meter"] != "ok") { printf "NO-TABLE: %s meter=%s, want ok — a failed wait or an out-of-band coverage is not a measurement\n", logname, (r["meter"] == "" ? "ABSENT" : r["meter"]); bad = 1 }
    if (r["coverage"] == "") { printf "NO-TABLE: %s RESULT line carries no coverage= field\n", logname; bad = 1 }
    else if (r["coverage"] + 0 < 0.8 || r["coverage"] + 0 > 1.2) { printf "NO-TABLE: %s coverage=%s is outside 0.8-1.2 — the phases do not account for the round window, so a phase is missing and this attribution is not readable\n", logname, r["coverage"]; bad = 1 }
    if (nrows == 0) { printf "NO-TABLE: %s printed no phase lines for role=%s — the meter never armed, so this binary does not contain the instrument the table is about\n", logname, role; bad = 1 }
    if (wall == 0 && r["wall_ms_per_round"] == "") { printf "NO-TABLE: %s carries no wall_ms_per_round= on its summary line, so nothing can be checked against the round's wall\n", logname; bad = 1 }
    if (bad) exit 1

    printf "  -- role=%s lanes=%s coverage=%s sum=%.3f wall=%.3f ms/round windows=%s syncs_added=%s idle_sync_us=%s plain_steps=%s sync_errors=%s\n", role, w, r["coverage"], r["sum_ms_per_round"] + 0, r["wall_ms_per_round"] + 0, ins["windows"], ins["syncs_added"], ins["idle_sync_us"], ins["plain_steps"], ins["sync_errors"]
    printf "   %-21s %12s %6s %11s %5s %12s\n", "phase", "ms_per_round", "runs", "ms_per_run", "sync", "status"
    sum = 0
    for (i = 1; i <= nrows; i++) {
        if (stv[i] == "not_metered") { printf "   %-21s %12s %6s %11s %5s %12s\n", name[i], "-", "-", "-", "-", "not_metered"; continue }
        printf "   %-21s %12.4f %6s %11.4f %5s %12s\n", name[i], prv[i] + 0, runsv[i], sprv[i] + 0, sav[i], (stv[i] == "" ? "measured" : stv[i])
        if (stv[i] != "derived") sum += prv[i] + 0
    }
    if ((sum + ov) - wall > 0.002 || wall - (sum + ov) > 0.002) {
        printf "NO-TABLE: sum(phases)=%.4f + overhead=%.4f != wall=%.4f — the meter is double-charging a phase or dropping one\n", sum, ov, wall
        exit 1
    }
    # The second check is the one that catches a missing ROW rather than a missing PHASE: `sum_ms_per_round`
    # comes from the engine's own accumulator and the sum above comes from the printed rows, so a phase that
    # the engine charged but `describe()` did not print moves one and not the other. Without it, the identity
    # check above passes vacuously whenever `overhead` absorbs whatever went missing.
    if (r["sum_ms_per_round"] == "") {
        printf "NO-TABLE: %s RESULT line carries no sum_ms_per_round= field to cross-check the rows against\n", logname
        exit 1
    }
    if ((r["sum_ms_per_round"] + 0) - sum > 0.002 || sum - (r["sum_ms_per_round"] + 0) > 0.002) {
        printf "NO-TABLE: the engine billed sum_ms_per_round=%s but the printed phase rows add to %.4f — a phase is charged and not printed\n", r["sum_ms_per_round"], sum
        exit 1
    }
    printf "   identity ok: sum(phases) %.4f + overhead %.4f == wall %.4f ms/round, and == RESULT sum_ms_per_round %s\n", sum, ov, wall, r["sum_ms_per_round"]
}
