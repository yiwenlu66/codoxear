# Browser driver notes (sanitized)

Session: `AGENT_BROWSER_SESSION=fakecc19397`
URL: `http://127.0.0.1:19397/`
Password was the Docker sandbox default and is not recorded here.

Flow exercised through the actual UI:
1. Open `/`, login.
2. Click New session.
3. Click Claude backend tab.
4. Fill working directory `/workspace`.
5. Fill session name `fake claude usable proof`.
6. Fill model `sonnet` and accept the combobox value.
7. Open reasoning effort picker and choose `max`.
8. Uncheck `Create in tmux` for direct launch.
9. Click Start session.
10. Select/use the bound session automatically projected at `#session=broker-190`.
11. Fill composer `#msg` with `USER_SENTINEL_FAKE_CC_PROMPT_19397 please answer through fake claude`.
12. Click `#sendBtn`.

Important eval probes are saved as `eval-before-start.json`, `eval-after-bind-browser-state.json`, and `eval-after-send-idle-browser-state.json`.
