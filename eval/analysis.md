### Strengths

- **File quality in images, audio, and video: 30 of 33 found.** Corrupt, blurry, dark, silent,
  clipped, too short, black, and frozen files are measured by code and removed by allowed
  operations, and the cleaning log names every removed file.
- **Duplicates were found in every dataset that had them**, exact and near.
- **Prompt injection was found, masked, reported, and not obeyed.** In the HR table, two
  cells told the AI to call the data clean; the scan found both, the agents reported them
  as a critical security risk, and the run still reported the real problems.
- **Numeric target leakage is caught** (the sales table's `refund_amount`: AUC 1.0), and so
  are cross-type problems in mixed zips (files without rows, labels that disagree with folders).

### Problems the evaluation found, fixed in version 0.10.0

1. **European number and date formats were silently corrupted.** The cleaning log showed
   `'2.167,60 EUR' → 2.1676`, `'15,0' → 150`, and `02.03.2025` read as 3 February. The
   conversions trusted the agent's `decimal_comma` and `dayfirst` options, and the agent
   often sent `false` as a default. Now the format is worked out from the values unless the
   agent explicitly asks for it, and the evaluation checks the cleaned values themselves
   (a step that runs but produces wrong values doesn't count). The European dataset scored
   3 of 6 before the fix (`eval/results/before-fix/`) and 5 of 6 after.
2. **The "numbers fact-checked" count missed numbers written only in the sentence** (the HR
   run shows 0, though its findings quote ages of -3 and 212). Every published number was
   checked; only the count was wrong. The other rows are from before this fix.
3. Renamed columns were described in the cleaning log as removed and added.

### Known gaps (not fixed in this version)

- **Missingness that depends on the target isn't checked.** HR's `exit_interview_score`
  exists only for people who left; the agents saw that it was 87% missing but didn't call
  it leakage. Only a numeric feature's AUC against the target is measured.
- **No per-group or physical-range checks:** a stuck sensor, one sensor switching to
  Fahrenheit, and humidity over 100% were all missed. The -999 code was only noticed by
  the reviewer.
- **Other misses:** the log's 40-minute silence was measured by code
  but not reported; the HR table's 9% minority class, ID column, and email addresses
  weren't flagged; nor were the music-only audio clip, the ticket filed under two labels,
  the very long ticket, and the video's missing audio track and rotation metadata.
- **Qualitative claims aren't fact-checked.** In every European run the analyst said text
  was "masked during cleaning" although the masking step changed nothing (the formula
  cell is neutralized when the CSV is exported).
- "k.A." (German for "not specified") isn't a known missing-value token, and
  `normalize_case` turned "HR" into "Hr".
- The fact check rejected the analyst's first findings in 7 of 10 runs. Retries fixed them,
  at the cost of extra model calls and time.
