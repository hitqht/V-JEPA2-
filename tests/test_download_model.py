import hashlib
import json
from pathlib import Path
import pytest
import requests
from scripts import download_model as d


def test_auto_falls_back_on_network_failure(tmp_path,monkeypatch):
    calls=[]
    def unavailable(out):raise requests.ConnectionError('offline')
    monkeypatch.setattr(d,'download_hf',unavailable)
    monkeypatch.setattr(d,'download_meta',lambda checkpoint,out:calls.append((checkpoint,out)))
    out=tmp_path/'weights';checkpoint=tmp_path/'source.pt'
    d.main(['--source','auto','--out',str(out),'--checkpoint',str(checkpoint)])
    assert calls==[(checkpoint,out)]


def test_explicit_hf_does_not_silently_switch_sources(tmp_path,monkeypatch):
    def unavailable(out):raise requests.ConnectionError('offline')
    monkeypatch.setattr(d,'download_hf',unavailable)
    with pytest.raises(requests.ConnectionError):d.main(['--source','hf','--out',str(tmp_path)])


def test_auto_reuses_hash_verified_weights(tmp_path,monkeypatch):
    monkeypatch.setattr(d,'ROOT',tmp_path)
    out=tmp_path/'weights';out.mkdir()
    payload=b'not-real-weights-only-for-testing'
    (out/'model.safetensors').write_bytes(payload)
    (out/'config.json').write_text('{}')
    (out/'download_manifest.json').write_text(json.dumps({'converted_weights_sha256':hashlib.sha256(payload).hexdigest()}))
    monkeypatch.setattr(d,'download_hf',lambda out:pytest.fail('must not download verified weights'))
    d.main(['--out',str(out)])
    assert (tmp_path/'reports/model_manifest.json').is_file()


def test_existing_corrupt_weights_fail_before_network(tmp_path,monkeypatch):
    (tmp_path/'model.safetensors').write_bytes(b'corrupt')
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'download_manifest.json').write_text(json.dumps({'converted_weights_sha256':'0'*64}))
    monkeypatch.setattr(d,'download_hf',lambda out:pytest.fail('must reject corrupt artifact'))
    with pytest.raises(ValueError,match='哈希'):d.main(['--out',str(tmp_path)])
