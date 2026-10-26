"""Execute quick-launch and full-dialog controllers against observable VM nodes."""
import json
import subprocess

from frontend_module_loader import module_path


SOURCES = {name: module_path(name).read_text() for name in (
    "app_launch.js", "app_display.js", "app_session_state.js", "app_session_catalog.js",
    "app_new_session.js", "app_quick_session.js",
)}
PRELUDE = r"""
const vm = require('vm'), assert = require('node:assert/strict');
const storage = {data:new Map(),getItem(k){return this.data.get(k)||null},setItem(k,v){this.data.set(k,String(v))},removeItem(k){this.data.delete(k)}};
const noop = ()=>{};
const ctx = {window:{CodoxearStorage:storage,CodoxearUrls:{resolveAppUrl:p=>p}}, console,
  HTMLElement: class {}, setTimeout:()=>1, clearTimeout:noop};
vm.createContext(ctx);
for (const [name,source] of Object.entries(SOURCES)) vm.runInContext(source,ctx,{filename:name});
const L = ctx.window.CodoxearLaunch;
const defaults = {default_backend:'pi',backends:{pi:{provider_choices:['p'],provider_choice:'p',model:'model-a',provider_models:{p:['model-a','model-b']},reasoning_effort:'high',reasoning_efforts:['low','high']},codex:{provider_choice:'chatgpt',model:'codex-model'}}};
function node(extra={}) {return {style:{},children:[],value:'',textContent:'',disabled:false,checked:false,
 classList:{toggle:noop,add:noop,remove:noop,contains:()=>false},setAttribute(k,v){this[k]=String(v)},removeAttribute:noop,
 appendChild(c){this.children.push(c);return c},insertBefore(c){this.children.unshift(c)},focus(){this.focused=true},
 contains(n){return n===this},querySelector:()=>null,querySelectorAll:()=>[],setSelectionRange:noop,...extra};}
const nodes = new Map();
const el = (tag,attrs={},children=[])=>{const n=node({tag});for(const [k,v] of Object.entries(attrs)){if(k==='text')n.textContent=v;else n[k]=v} n.children=children; if(n.id)nodes.set(n.id,n);return n};
const state = ctx.window.CodoxearSessionState.createSessionState({consoleError:noop});
const catalog = ctx.window.CodoxearSessionCatalog.createSessionCatalog({consoleError:noop});
catalog.set('newSessionDefaults',defaults);catalog.set('tmuxAvailable',true);
const doc = {activeElement:null,createElement:()=>node(),body:node(),getElementById:id=>nodes.get(id),querySelectorAll:()=>[]};
const events=[];
const eventBindings={on(target,type,handler){events.push({target,type,handler})}};
const record={backend:'pi',providerChoice:'p',model:'model-b',providerAbsent:false,reasoningEffort:'low',fast:false,tmux:false};
function quickHarness({result=123,error=null}={}) {
 let resolveSpawn, spawnArgs, dialog, selections=[], restores=[], blocked=false;
 const text=node(), copy=node({textContent:'Idle'});
 const controller=ctx.window.CodoxearQuickSession.createQuickSessionController({
  el,iconSvg:()=>'',sidebarHeaderActions:node(),emptyActions:node(),emptyCopy:copy,textarea:text,
  sessionState:state,sessionCatalog:catalog,eventBindings,document:doc,
  deselectSession(){state.set('selected',null);text.value=''},
  selectSession(id){state.set('selected',id);text.value=storage.getItem('draft.'+id)||'';selections.push(id);return Promise.resolve()},
  spawnSession(...args){spawnArgs=args;return new Promise(resolve=>{resolveSpawn=resolve})},
  restoreDraft(id,value){if(!text.value){text.value=value;restores.push([id,value])}},
  openDialog(o){dialog=o},closeSidebar:noop,shortcutBlocked:()=>blocked,
 });
 return {controller,text,copy,get args(){return spawnArgs},get dialog(){return dialog},selections,restores,
  async resolve({ready=true}={}){if(error)spawnArgs[9]({message:error});if(ready&&result)await spawnArgs[11]('new');resolveSpawn(result)},
  block(v){blocked=v}};
}
function dialogHarness() {
 let result=null, args;
 const controller=ctx.window.CodoxearNewSession.createNewSessionDialogController({
  root:node(),el,iconSvg:()=>'',document:doc,window:{requestAnimationFrame:noop},addEvent:noop,
  sessionState:state,sessionCatalog:catalog,isMobile:()=>false,prepareModalOpen:noop,
  afterModalVisibilityChanged:noop,isModalTargetOpen:n=>n.style.display!=='none',applyDialogMenus:noop,
  positionDialogMenu:noop,setPickerButtonContent:noop,fetchResumeCandidates:async()=>({sessions:[]}),
  spawnSession:async(...a)=>{args=a;return result},
 });
 return {controller,setResult(v){result=v},get args(){return args}};
}
const tick=()=>new Promise(r=>setImmediate(r));
"""


def run(body):
    script = "const SOURCES=" + json.dumps(SOURCES) + ";\n" + PRELUDE
    script += "\n(async()=>{" + body + "\nconsole.log(JSON.stringify({ok:true}));})().catch(e=>{console.error(e);process.exit(1)});"
    proc = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, timeout=20)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout)


def test_last_launch_record_only_changes_after_successful_dialog_start():
    run(r"""
const h=dialogHarness(); h.controller.open({cwd:'~'});
assert.equal(storage.getItem('codoxear.lastLaunch'),null);
const tabs=nodes.get('newSessionBackendTabs');
// Actual backend tab buttons carry their click handlers, just like the browser.
for(const child of [...tabs.children]) if(child.onclick)child.onclick();
assert.equal(storage.getItem('codoxear.lastLaunch'),null);
h.controller.close(); h.controller.open({cwd:'~',launchPreset:record});
nodes.get('newSessionStartBtn').onclick(); await tick();
assert.equal(storage.getItem('codoxear.lastLaunch'),null);
h.setResult('launch-pending-id'); nodes.get('newSessionStartBtn').onclick(); await tick();
assert.deepEqual(JSON.parse(storage.getItem('codoxear.lastLaunch')),record);
h.controller.open({cwd:'/tmp'});
assert.deepEqual(JSON.parse(storage.getItem('codoxear.lastLaunch')),record);
assert.notEqual(nodes.get('newSessionReasoningBtn').textContent,'low');
""")


def test_record_and_fallback_use_dialog_defaults_and_remembered_pair():
    run(r"""
const plain=L.loadLastLaunch(defaults,true);
assert.equal(plain.backend,'pi');assert.equal(plain.model,'model-a');assert.equal(plain.reasoningEffort,'high');assert.equal(plain.tmux,true);
L.rememberProviderModelChoice('pi','p','model-b');
assert.equal(L.loadLastLaunch(defaults,true).model,'model-b');
L.rememberProviderModelChoice('pi','','org/model',{providerAbsent:true});
const absent=L.loadLastLaunch(defaults,false);assert.equal(absent.model,'org/model');assert.equal(absent.providerChoice,'');assert.equal(absent.providerAbsent,true);
L.rememberLastLaunch(record);assert.deepEqual(JSON.parse(JSON.stringify(L.loadLastLaunch(defaults,true))),record);
L.rememberLastLaunch({...record,backend:'unknown'});assert.equal(L.loadLastLaunch(defaults,false).backend,'pi');
storage.setItem('codoxear.lastLaunch','bad json');assert.equal(L.loadLastLaunch(defaults,false).backend,'pi');
""")


def test_quick_start_focus_cwd_repeat_guard_and_preselection_draft_capture():
    run(r"""
L.rememberLastLaunch(record);state.set('selected','old');const h=quickHarness();
const launch=h.controller.start();h.controller.start();
assert.equal(state.get('selected'),null);assert.equal(h.text.focused,true);assert.equal(h.controller.isStarting(),true);
assert.equal(h.args[0],'~');assert.equal(h.args[1],null);assert.equal(h.args[4],'p');assert.equal(h.args[5],'model-b');assert.equal(h.args[6],'low');assert.equal(h.args[8],false);
assert.equal(h.copy.textContent,'Starting Pi · model-b · low in ~…');
h.text.value='Typed during startup';await h.resolve();await launch;
assert.deepEqual(h.restores,[['new','Typed during startup']]);assert.equal(h.text.value,'Typed during startup');assert.equal(h.controller.isStarting(),false);
assert.deepEqual(JSON.parse(storage.getItem('codoxear.lastLaunch')),record);
""")


def test_nonempty_target_draft_wins_and_manual_selection_cancels_handoff():
    run(r"""
const h=quickHarness();const launch=h.controller.start();h.text.value='scratch';storage.setItem('draft.new','existing');
await h.resolve();await launch;assert.equal(h.text.value,'existing');assert.deepEqual(h.restores,[]);
h.controller.dispose();const other=quickHarness();const next=other.controller.start();other.text.value='do not move';
state.set('selected','manual');assert.equal(other.controller.isStarting(),false);
await other.resolve();await next;assert.equal(state.get('selected'),'manual');assert.deepEqual(other.selections,[]);assert.deepEqual(other.restores,[]);
""")


def test_failure_retains_text_and_opens_preset_dialog():
    run(r"""
L.rememberLastLaunch(record);const h=quickHarness({result:null,error:'Provider unavailable'});const launch=h.controller.start();
h.text.value='keep me';await h.resolve();await launch;
assert.equal(h.text.value,'keep me');assert.equal(h.controller.isStarting(),false);assert.equal(h.dialog.cwd,'~');assert.equal(h.dialog.statusText,'Provider unavailable');
assert.deepEqual(JSON.parse(JSON.stringify(h.dialog.launchPreset)),record);
""")


def test_pending_launch_matches_catalog_and_failed_launch_opens_dialog():
    run(r"""
const h=quickHarness({result:'launch-1'});const launch=h.controller.start();h.text.value='pending text';await h.resolve({ready:false});await launch;
assert.equal(h.controller.isStarting(),true);catalog.set('latestSessions',[{session_id:'waiting',launch_id:'launch-1',launch_state:'starting'}]);assert.equal(h.controller.isStarting(),true);
catalog.set('latestSessions',[{session_id:'new',launch_id:'launch-1'}]);await tick();assert.equal(h.text.value,'pending text');assert.equal(state.get('selected'),'new');
h.controller.dispose();const f=quickHarness({result:'launch-2'});const failed=f.controller.start();await f.resolve({ready:false});await failed;
catalog.set('latestSessions',[{session_id:'failed',launch_id:'launch-2',launch_state:'failed',launch_error:'Unavailable'}]);assert.equal(f.dialog.statusText,'Unavailable');
""")


def test_shift_n_respects_text_entry_modal_and_modifier_guards():
    run(r"""
const h=quickHarness();const handler=events.find(e=>e.target===doc&&e.type==='keydown').handler;
h.block(true);handler({key:'N'});assert.equal(h.controller.isStarting(),false);
h.block(false);handler({key:'N',ctrlKey:true});assert.equal(h.controller.isStarting(),false);
let prevented=false;handler({key:'N',preventDefault(){prevented=true}});assert.equal(prevented,true);assert.equal(h.controller.isStarting(),true);
await h.resolve();
""")
