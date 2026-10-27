# Second Brain "Ask" answers: pass/fail rubric (v1.1)

Used for both the human labels and the pointwise judge prompt.

## The question
Given **only the retrieved notes**, is this the answer the user should have got?

Each item shows: the question, the retrieved notes (numbered in retrieval order, with dates), and the answer.

## PASS (all must hold)
1. **Supported**: every factual claim about the user can be traced to a retrieved note.
   Paraphrase and light summarising are fine.
2. **Answers the question**: its main content addresses what was asked.
3. **Citations, if present, are right**: a cited note contains what is claimed.
   Citations can be `[n]` (= retrieved note n) or by date ("in your note from Nov 13, 2025");
   a date citation points to the note(s) with that date. No citations at all is not a fail on its own.
4. **Declines when it should**: if the notes don't support an answer, it says so clearly.

## FAIL (any one is enough)
| Tag | What it looks like |
|---|---|
| `unsupported` | A claim not in the notes (invented, general knowledge stated as the user's, or a note described as something it isn't, e.g. an ordinary note presented as "inspirational"). |
| `should-abstain` | The premise isn't supported (e.g. "three things I do at night" with no such notes) but it answers as if it were. |
| `false-abstain` | The notes clearly answer the question, but it declines. |
| `wrong-citation` | A cited note (by number or date) doesn't support the claim attached to it. |
| `off-topic` | Its main content addresses something other than what was asked (e.g. asked about recipes, summarises mood and learning), or the actual answer is buried under unrelated content. |

A partly right answer with one unsupported claim is a FAIL (`unsupported`).

## Rules for common edge cases
- **Lead with the answer.** The answer, or the statement that the notes don't support one, must come
  first. FAIL if it opens with unrelated content and the answer only appears as a side point
  (`off-topic`, e.g. asked for wins, leads with struggles), or opens with a claim that the later text
  contradicts (`should-abstain` / `unsupported`, e.g. answers first, admits "not in your notes" at the end).
  Extra content after a clear answer = PASS (`extra`).
- **Hedge, then suggest → PASS.** It clearly says the notes don't answer the question, and what follows
  is supported and framed as a suggestion ("tools you plan to check..."). If after the hedge it answers
  the premise as if true → FAIL `should-abstain`.
- **True but irrelevant extra content → PASS** when the main question is answered and nothing is false.
  Extra material next to a real answer is not `off-topic`. Note it with `extra` or `borderline`.

## Still open
- Wrong or approximate dates.
- Time words in the question ("today", "lately"): the label assumes the date the question was asked.

## Label format
`label`: `pass` | `fail`.
`note`: for fails, a tag plus a few words on why (`off-topic - summarised mood instead of recipes`).
For passes, optional: `hedge`, `extra`, `borderline` plus a few words.

## Changelog
- v1.1 (2026-10-27): during batch 2. "Lead with the answer" rule, written down after labels already
  applied it (items 11, 25, 46); `off-topic` covers buried answers. No v1 label changed.
- v1 (2026-10-27): after batch 1 (10 items). Hedge-then-suggest = pass; true-but-irrelevant extra = pass;
  `off-topic` limited to main content; date-style citations count; mischaracterised notes = `unsupported`;
  fail notes say why.
- v0 (2026-10-27): first draft, before any labelling.
