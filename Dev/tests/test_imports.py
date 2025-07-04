import importlib

modules = [
    'study_tools.build_index',
    'study_tools.summarize',
    'study_tools.cli_chat',
    'study_tools.flashcards',
    'study_tools.ingest',
    'study_tools.reset',
    'study_tools.utils',
]

def test_imports():
    for m in modules:
        importlib.import_module(m)
