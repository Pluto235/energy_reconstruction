"""Strict native ROOT header read, with independent evidence comparison."""
from array import array
import hashlib
import json

FIELDS = ['irun', 'ies', 'iseq', 'poolused']
INTEGER_TYPES = {'Int_t', 'UInt_t', 'Long64_t', 'ULong64_t', 'Short_t', 'UShort_t'}


def validate_evidence(actual, expected):
    if (actual['entries'] != 1 or actual['header_keys'] != 1 or
            actual['read_bytes'] <= 0 or not actual['uuid'] or
            any(actual['types'][k] not in INTEGER_TYPES for k in FIELDS)):
        raise ValueError('native header is not unique/readable/integer')
    if actual['uuid'] != expected['uuid'] or actual['header'] != expected['header']:
        raise ValueError('native header differs from independent UUID/identity/pool evidence')
    if not 0 <= actual['header']['poolused'][0] <= 7:
        raise ValueError('native header invalid pool mask')


def inspect(ROOT, path, expected):
    f = ROOT.TFile.Open(path, 'READ')
    if not f or f.IsZombie():
        raise ValueError('native ROOT cannot open header')
    try:
        if f.TestBit(ROOT.TFile.kRecovered):
            raise ValueError('native ROOT recovered file is not trusted')
        count = sum(key.GetName() == 'wcdah' for key in f.GetListOfKeys())
        if count != 1:
            raise ValueError('native header key is not unique')
        tree = f.Get('wcdah')
        if not tree or not tree.InheritsFrom('TTree') or tree.GetEntries() != 1:
            raise ValueError('native header tree is missing/non-unique')
        types = {}
        tree.SetBranchStatus('*', 0)
        for key in FIELDS:
            leaf = tree.GetLeaf(key)
            if not leaf or leaf.GetLenStatic() != 1 or leaf.GetLeafCount():
                raise ValueError('native header field is missing or not scalar: '+key)
            types[key] = str(leaf.GetTypeName())
            if types[key] not in INTEGER_TYPES:
                raise ValueError('native header field is not integer: '+key)
            tree.SetBranchStatus(key, 1)
        size = int(tree.GetEntry(0))
        actual = dict(path=path, uuid=str(f.GetUUID().AsString()), entries=1,
                      header_keys=count, read_bytes=size, types=types,
                      header={key: [int(getattr(tree, key))] for key in FIELDS},
                      reader='native_ROOT', root_version=str(ROOT.gROOT.GetVersion()))
        validate_evidence(actual, expected)
        actual['header_sha256'] = hashlib.sha256(json.dumps(actual['header'], sort_keys=True).encode()).hexdigest()
        return actual
    finally:
        f.Close()


def selfcheck(ROOT, folder):
    path = folder/'native-header-selfcheck.root'
    f = ROOT.TFile(str(path), 'RECREATE'); tree = ROOT.TTree('wcdah', 'wcdah')
    values = {k: array('i', [v]) for k, v in zip(FIELDS, [123, 2, 456, 3])}
    for key, value in values.items(): tree.Branch(key, value, key+'/I')
    tree.Fill(); tree.Write()
    expected = dict(uuid=str(f.GetUUID().AsString()), header={k: list(v) for k, v in values.items()})
    f.Close()
    result = inspect(ROOT, str(path), expected)
    path.unlink()
    return dict(passed=True, non7_pool_preserved=result['header']['poolused'][0] == 3)
