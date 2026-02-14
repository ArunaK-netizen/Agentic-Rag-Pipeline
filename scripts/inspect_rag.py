import importlib, sys, os, inspect
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    m = importlib.import_module('src.rag_pipeline')
    print('module file:', getattr(m, '__file__', None))
    print('\n--- process_documents start ---\n')
    print(inspect.getsource(m.RAGPipeline.process_documents))
    print('\n--- process_documents end ---\n')
except Exception as e:
    print('inspect error:', e)
