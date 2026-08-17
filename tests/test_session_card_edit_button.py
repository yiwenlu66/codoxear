"""Behavioral test: sidebar session card renders the rename (Edit conversation) button
for normal (non-failed, non-pending, non-lost) sessions and clicking it invokes openEditSession.
Uses Node VM to execute the real shipped modules.
"""
import json, subprocess, sys, tempfile, textwrap
from frontend_module_loader import module_path
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

JS = textwrap.dedent("""
    const vm = require('vm');
    const helpers = __HELPERS__;
    const sessions = __SESSIONS__;

    function el(tag, attrs, children) {
      const node = { tag, attrs: attrs || {}, children: [], dataset: {}, style: {},
        text: attrs && attrs.text != null ? attrs.text : undefined,
        appendChild(c){ this.children.push(c); return c; },
        find(pred){ return this.children.find(pred) || null; },
        classList: { add: () => {}, remove: () => {} },
      };
      if (Array.isArray(children)) children.forEach(c => node.appendChild(c));
      return node;
    }
    const iconSvg = () => '<svg/>';

    const ctx = {
      window: {}, document: {},
      requireFunction: (fn, name) => { if (typeof fn !== 'function') throw new Error('missing '+name); return fn; },
      __editOpened: false,
    };
    vm.createContext(ctx);
    vm.runInContext(helpers, ctx);
    vm.runInContext(sessions, ctx);

    const noop = () => {};
    const sessionsWrap = el('div');
    const sessionState = { get: () => null, subscribe: () => () => true };
    const deps = {
      sessionState, sessionsWrap,
      sidebarEmptyHint: el('div'),
      el, iconSvg,
      agentBackendDisplayName: noop, agentBackendLogoPath: noop,
      api: noop, baseName: (path) => String(path).split('/').filter(Boolean).pop() || '', clearDeletedSessionClientState: noop,
      confirmAction: () => Promise.resolve(true), duplicateSession: noop,
      fmtRelativeAge: () => '1m ago', openEditSession: () => { ctx.__editOpened = true; },
      redactedLaunchErrorText: noop, refreshSessions: noop, selectSession: noop,
      sessionAgentBackend: noop, sessionDisplayName: noop,
      sessionIsFast: () => false,
      sessionLaunchFailed: (s) => !!(s && (s.lost || String(s.launch_state||'').toLowerCase()==='failed')),
      sessionLaunchIcon: noop, sessionLaunchLabel: noop, sessionLaunchPending: noop,
      setSidebarOpen: noop, setToast: noop,
      sidebarEffortCode: () => 'hi', sidebarModelText: () => 'kimi-k3',
      sidebarRenderSignature: () => 'sig',
      sidebarSessionEntries: (ss) => ss.map(s => ({type:'session', session:s})),
    };
    const ctrl = ctx.window.CodoxearSessions.createSessionsController(deps);

    const cases = [
      { label: 'normal busy session', session: { session_id: 's1', busy: true, cwd: '/x', git_branch: 'main', agent_backend: 'pi', launch_state: null, lost: false } },
      { label: 'idle session', session: { session_id: 's2', busy: false, cwd: '/x', git_branch: 'main', agent_backend: 'pi', launch_state: null, lost: false } },
    ];
    function findEditBtn(node) {
      if (!node) return null;
      if (node.attrs && node.attrs['aria-label'] === 'Edit conversation') return node;
      for (const c of node.children || []) { const r = findEditBtn(c); if (r) return r; }
      return null;
    }
    function nodesWithClass(node, className, out = []) {
      if (!node) return out;
      if (node.attrs && String(node.attrs.class || '').split(/\\s+/).includes(className)) out.push(node);
      for (const c of node.children || []) nodesWithClass(c, className, out);
      return out;
    }
    const results = [];
    for (const c of cases) {
      ctx.__editOpened = false;
      sessionsWrap.children = [];
      ctrl.renderSessions([c.session], { selectedId: '', swipeActions: false });
      const btn = findEditBtn(sessionsWrap);
      let opened = false;
      if (btn) btn.onclick({ preventDefault(){}, stopPropagation(){} });
      opened = ctx.__editOpened;
      results.push({
        label: c.label,
        hasBtn: !!btn,
        clickOpensEditor: opened,
        metadataDataSegments: nodesWithClass(sessionsWrap, 'sidebarMetaData').length,
        metadataLabelSegments: nodesWithClass(sessionsWrap, 'sidebarMetaLabel').length,
      });
    }
    console.log(JSON.stringify(results));
    """)


JS = JS.replace("__HELPERS__", json.dumps(module_path("app_session_helpers.js").read_text(encoding="utf-8"))).replace("__SESSIONS__", json.dumps(module_path("app_sessions.js").read_text(encoding="utf-8")))


class TestSidebarEditButton:
    def test_edit_button_renders_and_opens_editor(self):
        with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
            f.write(JS)
            js_path = f.name
        try:
            result = subprocess.run(['node', js_path], capture_output=True, text=True, cwd=REPO, timeout=15)
            assert result.returncode == 0, f"node failed: {result.stderr}"
            data = json.loads(result.stdout.strip().splitlines()[-1])
            assert len(data) == 2, f"unexpected result count: {data}"
            for case in data:
                assert case['hasBtn'], f"edit button missing for {case['label']}"
                assert case['clickOpensEditor'], f"click did not open editor for {case['label']}"
            assert case['metadataDataSegments'] == 1, f"model/effort should be the sole data segment for {case['label']}"
            assert case['metadataLabelSegments'] == 3, f"state, cwd, and branch should remain proportional labels for {case['label']}"
        finally:
            Path(js_path).unlink(missing_ok=True)
