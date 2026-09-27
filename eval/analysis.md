These results are from version 0.10.1. The first published run (version 0.10.0, 68 of 83)
is kept in `eval/results/v0.10.0/`. Model output varies from run to run, so small
differences between the two runs aren't meaningful on their own.

### Strengths

- **File quality in images, audio, and video: 29 of 33 found.** Corrupt, blurry, silent,
  clipped, too short, black, and frozen files are measured by code and removed by allowed
  operations, and the cleaning log names every removed file.
- **Logs: 4 of 4.** The timeline check found the 40-minute silence and the error burst
  (34 errors in 7 minutes, all from the payment service), and the agents reported both.
- **Mixed data: 6 of 6.** Code linked the table to the files and found every planted
  mismatch; the agents reported all three kinds.
- **Duplicates were found in every dataset that had them.**
- **Prompt injection was found, masked, reported, and not obeyed.** In the HR table, two
  cells told the AI to call the data clean; the scan found both, the agents reported them,
  and the run still reported the real problems.
- **Numeric target leakage is caught** (the sales table's `refund_amount`: AUC 1.0).

### Problems the evaluation found, fixed before these results

1. **European number and date formats were silently corrupted** in version 0.10.0: the
   cleaning log showed `'2.167,60 EUR' → 2.1676`, `'15,0' → 150`, and `02.03.2025` read as
   3 February. The conversions trusted the agent's `decimal_comma` and `dayfirst` options,
   and the agent often sent `false` as a default. Now the format is worked out from the
   values, and only an explicit `true` overrides it. The evaluation checks the cleaned
   values themselves: the European dataset scored 3 of 6 before the fix
   (`eval/results/before-fix/`) and 5 of 6 after.
2. **The "numbers fact-checked" count missed numbers written only in the sentence.**
   Every published number was checked; only the count was wrong.
3. **Reading the agents' messages turned up two behavior problems** (version 0.10.1): the
   strategist filled a mostly empty free-text column with its most common value (inventing
   16 notes), and triage picked a file-name column as the target. The dry run now refuses to
   fill in more than 40% of a column, and triage can't pick an identifier as the target.

### Known gaps

- **Target-dependent missingness isn't recognized as leakage.** HR's
  `exit_interview_score` exists only for people who left. The strategist dropped it, but
  for being 87% missing, not as a leak, so it counts as fixed here without being
  understood. Only a numeric feature's AUC against the target is measured.
- **No per-group checks:** a stuck sensor and one sensor switching to Fahrenheit were
  missed. The -999 code and humidity over 100% were caught only by the reviewer's notes.
- **Other misses:** mixed date formats in the sales table (the strategist didn't parse
  them this run), "k.A." as a missing-value marker, dark and blank images (removed as blurry,
  never named), a music-only audio clip, near-duplicate and very long support tickets, a
  video with no audio track, rotation metadata, and the HR table's email addresses.
- **Qualitative claims aren't fact-checked.** The run described a spreadsheet formula as
  a "SQL/command injection string" that "was masked"; it was neutralized on export, not by
  a cleaning step.
- **`normalize_case` turned "HR" into "Hr".**
- **The fact check rejected the analyst's first findings in 6 of 10 runs.** Retries fixed
  them, at the cost of extra model calls and time.
