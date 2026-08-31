---
name: pr-reviewer
description: Review a pull request, branch, or the current branch's diff against six ordered priorities - correctness, extensibility, simplicity, readability, well-supported libraries, no reinvented wheels. Use when the user asks to review a PR, review their branch or changes, pastes a PR number or GitHub PR URL, or invokes /pr-reviewer. Supports --comment to post findings as inline GitHub PR comments and --out to save the report to a file.
---

# PR reviewer

Review a diff against the six ordered priorities below and print a ranked terminal
report. With `--comment` and a PR target, also post the findings as one inline GitHub
review.

## Resolving the target

Parse the arguments (excluding `--comment` and `--out <file>`):

- **PR number or GitHub PR URL** → fetch with `gh pr diff <n>` and `gh pr view <n>`.
  If `gh` is unavailable or unauthenticated, fall back to the GitHub MCP
  `pull_request_read` tool.
- **Branch name** → `git diff origin/<main>...<branch>`.
- **No target** → `git diff origin/<main>...HEAD`.

Detect the main branch (`git remote show origin` or `refs/remotes/origin/HEAD`) rather
than assuming it is named `main`. Run `git fetch origin` first — diffing against a stale
local main surfaces already-merged code as findings.

**Never review from the diff alone.** For each changed file, read the full file (or at
least the enclosing functions/classes) — most correctness and reinvention findings live
in the surrounding context, not the hunk. Also skim the repo for existing helpers that
the new code may duplicate.

**Large PRs:** over ~15 changed files, don't read everything serially into one context —
fan out per-file (or per-directory) review to subagents and merge their findings, or at
minimum order the reading by diff size and say which files got only a diff-level pass.

## Review priorities (in order)

Findings are ranked strictly by this order: every correctness finding outranks every
extensibility finding, and so on. Flag nothing outside these six categories — no style
nitpicks, no formatting, no bikeshedding.

### 1. Correctness

Logic errors, unhandled edge cases (empty input, None/null, off-by-one, error paths),
broken API/contract expectations, concurrency and state bugs, wrong behavior versus the
PR's stated intent. Before reporting a suspected bug, trace the actual code path and
confirm it — a plausible-looking bug that doesn't reproduce is worse than no finding.

**Breaking changes and compatibility** are correctness findings: API contract changes
that break callers outside the diff, DB schema migrations without a rollback path,
serialization-format changes that break already-stored data, renamed config keys or env
vars existing deployments rely on. These live in the callers and consumers the diff
doesn't touch — check them explicitly, don't wait to stumble on them.

### 2. Extensibility

Flag rigid code where variation **already exists or is explicitly named in the PR**:
copy-pasted branches that should be an abstract method, a mixin, or a strategy;
switch-on-type chains that a subclass hook would absorb; hardcoded values a second
existing caller already needs to differ.

Do NOT demand speculative abstractions for variation that might exist someday — that
violates priority 3. The bar: point to the second concrete case that the abstraction
would serve today. No second case, no finding.

### 3. Simplicity

Convoluted control flow, indirection layers with a single user, config for values that
never vary, hoop-jumping a direct implementation avoids. If a function can lose a layer
and do the same job, say so and show the shorter shape.

### 4. Readability

Misleading or vague names, functions doing three unrelated things, and comments that
compensate for unreadable code — the fix is renaming/restructuring the code and deleting
the comment, not improving the comment. Also flag comments that merely restate the code.
Good code with few comments beats bad code with many.

### 5. Well-supported libraries

Prefer maintained, widely-used packages. Flag: hand-rolled versions of what the
language's stdlib or an already-installed dependency provides; new dependencies on
obscure or unmaintained packages when a well-supported equivalent exists; pinning to a
deprecated API when the maintained replacement is drop-in.

Dependency hygiene belongs here too: a lockfile out of sync with the manifest (or a
manifest change with no lockfile update), a new dependency whose license conflicts with
the project's, and a version bump crossing a major with breaking changes the diff
doesn't account for.

### 6. No reinvented wheels

Custom retry loops, caches, date/duration math, path handling, serialization, or parsing
that duplicate stdlib, an existing dependency, or an existing helper elsewhere in the
same repo. Name the exact replacement (module/function or repo file) in the finding.

## Also flag (unranked, reported after the six categories)

Two checks outside the ranked categories, reported under a short "Also flag" section
(same finding format, usually low/medium severity):

- **PR-intent mismatch / stowaway changes**: unrelated refactors, drive-by formatting,
  or a second feature smuggled into a diff whose stated intent is something else. Name
  the stowaway and suggest splitting it out.
- **Leftover artifacts**: debug prints/console.logs, commented-out code, TODOs
  introduced by this PR, and dead code the change orphans.

## Report format

One line of context first (target, files changed), then findings ranked by category:

```
### 1. Correctness
- `path/file.py:42` **[high]** Off-by-one in pagination: last page dropped when
  `total % page_size == 0`. Fix: use `ceil(total / page_size)`.

### 2. Extensibility
- No findings.
...
```

Each finding: `file:line`, severity (**high** = must fix before merge, **medium** =
should fix, **low** = worth knowing), one-sentence issue, one-sentence concrete fix.
State "No findings." for empty categories — never invent filler to fill a section.
Close with a one-line verdict: merge-ready, merge after high-severity fixes, or needs
rework. The verdict keys off severity across ALL categories — a high-severity finding in
category 6 blocks merge just as much as one in category 1; the category order only ranks
the report, not the verdict.

With `--out <file>` (or when the user asks for a file), also write the report to that
file (default `pr<N>-review.md` for a PR target).

## --comment flag

Only valid with a PR target (number/URL) — with a branch or local diff, stop and tell
the user there is no PR to comment on.

Before posting, check for an existing review by the current user on this PR
(`pull_request_read` reviews): if one exists, don't post a duplicate — report the
findings in the terminal and tell the user a review already exists, unless they
explicitly asked to post again.

Post all findings as ONE review, not individual comments: GitHub MCP
`pull_request_review_write` with method `create` (pending review) →
`add_comment_to_pending_review` once per finding → `pull_request_review_write` with
method `submit_pending` (event COMMENT). Still print the terminal report as well.

**Anchoring:** GitHub only accepts inline comments on lines present in the diff. Anchor
each finding to its exact line when that line is in the diff; otherwise anchor to the
nearest changed line in the same file and name the real location in the comment text. A
finding in a file the PR doesn't touch goes in the review body, not inline. Never let an
anchoring error abandon a pending review — fall back to the review body and submit.

Posting a review is outward-facing: show the terminal report and confirm with the user
before submitting, unless they already told you to post without asking.
