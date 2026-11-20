from test_session_catalog import evaluate


def test_external_catalog_is_indexed_but_not_broker_catalog():
    result = evaluate('''
const catalog = createSessionCatalog({consoleError:()=>{},storage:null});
let seen=[];
catalog.subscribe('sessionIndex',()=>seen.push([...catalog.get('sessionIndex').keys()]));
catalog.applySnapshot({latestSessions:[{session_id:'broker'}],externalSessions:[{session_id:'outside',external:true}]});
catalog.patchSession('outside',{busy:true});
process.stdout.write(JSON.stringify({
 broker:catalog.get('latestSessions').map(s=>s.session_id),
 external:catalog.get('externalSessions'),seen
}));
''')
    assert result["broker"] == ["broker"]
    assert result["external"] == [{"session_id": "outside", "external": True, "busy": True}]
    assert result["seen"] == [["broker", "outside"], ["broker", "outside"]]


def test_external_removal_rebuilds_lookup_without_touching_broker_rows():
    result = evaluate('''
const catalog = createSessionCatalog({consoleError:()=>{},storage:null});
catalog.applySnapshot({latestSessions:[{session_id:'broker'}],externalSessions:[{session_id:'outside',external:true}]});
catalog.set('externalSessions',[]);
process.stdout.write(JSON.stringify({ids:[...catalog.get('sessionIndex').keys()],broker:catalog.get('latestSessions').length}));
''')
    assert result == {"ids": ["broker"], "broker": 1}
