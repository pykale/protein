import pytest
from kaleprotein.core.registry.base import Registry

def test_registry_register_get_available_and_missing():
    r=Registry('thing')
    class A: pass
    r.register('a', A)
    assert r.get('a') is A
    assert r.available_keys()==['a']
    with pytest.raises(KeyError): r.get('missing')

def test_registry_decorator():
    r=Registry('thing')
    @r.register(('x','y'))
    class B: pass
    assert r.get(('x','y')) is B


def test_namespaced_registry_resolves_case_insensitive_aliases():
    registry = Registry("component", normalize_strings=True)
    registry.register("sequence/cnn", object(), aliases=("Protein/CNN",))
    assert registry.get("protein/cnn") is registry.get("SEQUENCE/CNN")
