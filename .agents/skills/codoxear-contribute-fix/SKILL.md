---
name: codoxear-contribute-fix
description: Submit a Codoxear fix or feature as a real pull request to the canonical upstream yiwenlu66/codoxear. Use when the user asks to submit a fix, open a PR, contribute a change, or send work upstream. Covers duplicate checks, fork mechanics, scoped git branches, Docker-only verification, and exact gh PR creation.
---

# Contribute a Codoxear fix

The contribution target is the canonical upstream **yiwenlu66/codoxear** —
explicit and independent of this checkout's fork origin. Every `gh` call passes
`--repo yiwenlu66/codoxear`. A remote whose URL is `yiwenlu66/codoxear`
(HTTPS or SSH) is canonical; if none exists, add one:
`git remote add upstream https://github.com/yiwenlu66/codoxear.git`.

## 1. Authenticate and pick the push path

```sh
gh auth status        # `gh auth login` (device flow) when unauthenticated
gh api user -q .login
gh repo view yiwenlu66/codoxear --json viewerPermission -q .viewerPermission
```

- `WRITE`, `MAINTAIN`, or `ADMIN`: branch and push on the canonical remote.
- `READ`, `TRIAGE`, or `NONE`: fork once, then add it as a remote —

```sh
gh repo fork yiwenlu66/codoxear --clone=false
git remote add fork https://github.com/<your-login>/codoxear.git
```

## 2. Check for prior work

```sh
gh pr list --repo yiwenlu66/codoxear --search "<topic key words>" --state open --limit 20
gh issue list --repo yiwenlu66/codoxear --search "<topic key words>" --state all --limit 20
```

An open PR covering the change is a duplicate: extend or review it instead of
opening another. Note issues the work should close (`Fixes #N` in the body).

## 3. Scoped branch from current upstream main

```sh
git fetch <canonical-remote> main
git switch -c fix/<short-topic> <canonical-remote>/main   # or feature/<topic>
```

Change only what the fix needs: no drive-by refactors, reformatting, or
unrelated files. Follow the AGENTS.md contracts that touch the changed code.
Stage explicit files, review the staged diff, and commit scoped hunks with
messages that name the actual change.

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
gh pr create --repo yiwenlu66/codoxear \
  --head "<branch-owner>:fix/<short-topic>" \
  --title "<what changed>" \
  --body "<problem / change / verification evidence / Fixes #N>"
```

`--head owner:branch` when the branch lives on a fork; a plain branch name when
pushed to the canonical repo itself. Quote back only URLs that `git` and `gh`
actually printed.

## Boundaries

- Never create a PR or issue while testing this skill; validation is local
  discovery checks plus Docker verification only.
- Do not restart, deploy, or reconfigure a live server from a contribution
  session, and never push to `main` directly or force-push shared branches.
