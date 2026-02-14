import importlib, sys, os, inspect
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
try:
    m = importlib.import_module('src.pdf_processor')
    print('module file:', getattr(m, '__file__', None))
    print('has extract_text_from_image:', hasattr(m.PDFProcessor, 'extract_text_from_image'))
    print('\n--- Source start ---\n')
    src = inspect.getsource(m.PDFProcessor.extract_text_from_image)
    print(src)
    print('\n--- Source end ---\n')
except Exception as e:
    print('import/extract error:', e)
    try:
        print('\n--- process_uploaded_files source start ---\n')
        print(inspect.getsource(m.PDFProcessor.process_uploaded_files))
        print('\n--- process_uploaded_files source end ---\n')
    except Exception as e:
        print('process_uploaded_files inspect error:', e)
