---
name: codoxear-contribute-fix
description: Submit a Codoxear fix or feature as a real pull request to the canonical upstream yiwenlu66/codoxear. Use when the user asks to submit a fix, open a PR, contribute a change, or send work upstream. Covers duplicate checks, fork mechanics, scoped git branches, Docker-only verification, and exact gh PR creation.
---

# Contribute a Codoxear fix

The contribution target is the canonical upstream **yiwenlu66/codoxear** —
explicit and independent of this checkout's fork origin. Scope issue/PR commands
with `--repo yiwenlu66/codoxear`; repository commands take an explicit repository
argument. Auth commands and `gh api user` are account-scoped, not repo-scoped.
Identify the canonical git remote by its HTTPS or SSH URL, not its nickname.
If absent, add `https://github.com/yiwenlu66/codoxear.git` under an unused
remote name (for example `upstream`); never overwrite an existing remote.
Run the workflow from the editable checkout's root
(`cd "$(git rev-parse --show-toplevel)"`), or from the new worktree's root after
isolating a change there. Do not work in the serving snapshot.

## 1. Authenticate and pick the push path

```sh
gh auth status        # `gh auth login` (device flow) when unauthenticated
gh api user -q .login
gh repo view yiwenlu66/codoxear --json viewerPermission -q .viewerPermission
```

- `WRITE`, `MAINTAIN`, or `ADMIN`: branch and push on the canonical remote.
- `READ`, `TRIAGE`, or `NONE`: fork once, then add it as a remote —

```sh
gh repo fork yiwenlu66/codoxear --clone=false --remote=false
git remote add <unused-fork-remote> https://github.com/<your-login>/codoxear.git
```

## 2. Check for prior work

```sh
gh pr list --repo yiwenlu66/codoxear --search "<topic key words>" --state open --limit 20
gh issue list --repo yiwenlu66/codoxear --search "<topic key words>" --state all --limit 20
```

An open PR covering the change is a duplicate: extend or review it instead of
opening another. Note issues the work should close (`Fixes #N` in the body).

## 3. Preserve the fix and choose its branch

“Submit this fix” may refer to commits, staged changes, unstaged edits, or new
files already here. Inspect them before switching branches:

```sh
git status --short --branch
git diff
git diff --cached
git fetch <canonical-remote> main
git log --oneline <canonical-remote>/main..HEAD
git diff --stat <canonical-remote>/main...HEAD
```

- Already on a suitable topic branch containing only this change: keep it;
  review its full delta against upstream main before submission.
- The current HEAD contains only the intended commits (or equals upstream
  main), but needs a topic branch: `git switch -c fix/<short-topic>` at **HEAD**
  retains those commits and local edits. Do not start at upstream main and
  leave the existing fix behind.
- Mixed unrelated commits/edits, a shared branch, or a detached/serving
  snapshot: leave the original intact. Create a separate worktree from
  `<canonical-remote>/main` with `git worktree add -b fix/<short-topic>
  <new-worktree-path> <canonical-remote>/main`. Cherry-pick only the intended
  commits there, then transfer only this fix's staged/unstaged hunks and
  relevant untracked files, preserving their originals. Inspect the resulting
  diff; do not accidentally include unrelated work. If ownership of a hunk is
  genuinely unclear, ask which change belongs in the PR.
- No existing fix: create the scoped branch from fetched upstream main in a
  clean checkout or a separate worktree and implement it there.

Never reset, clean, blanket-stash, or overwrite user work to make a branch fit.
Follow the relevant AGENTS.md contracts. Stage explicit files or hunks, review
both the staged diff and the complete PR delta, and commit only this change
with a message that names it.

## 4. Verify in Docker only

Run the verification relevant to the change inside Docker:
`scripts/codoxear-docker-sandbox` (preflight/build/smoke/test), the focused
`scripts/docker_*_verify.sh` gates, or `scripts/docker_verify.sh [commit]`.
Per the repo testing policy: never against the live server (port 8743), never
on host runtime state. Do not deploy — `scripts/deploy.sh` is the maintainer's
post-merge step, not part of a contribution. Record which script ran and what
it observed; that evidence goes into the PR body.

## 5. Push and open the PR

```sh
git push -u <remote> fix/<short-topic>
gh pr create --repo yiwenlu66/codoxear --base main \
  --head "<branch-owner>:fix/<short-topic>" \
  --title "<what changed>" \
  --body "<problem / change / verification evidence / Fixes #N>"
```

`--head owner:branch` when the branch lives on a fork; a plain branch name when
pushed to the canonical repo itself. Substitute the actual retained topic
branch if reusing one. Quote back only URLs that `git` and `gh` actually printed.

## Boundaries

- Never create a PR or issue while testing this skill; validation is local
  discovery checks plus Docker verification only.
- Do not restart, deploy, or reconfigure a live server from a contribution
  session, and never push to `main` directly or force-push shared branches.
