import importlib, sys, os

# Ensure project root is on sys.path so `src` package is importable
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    m = importlib.import_module('src.gemini_vlm')
    print('module ok')
    try:
        print('api_key:', m._get_api_key())
    except Exception as e:
        print('api_key exc:', e)
    try:
        print('endpoint:', m._get_endpoint())
    except Exception as e:
        print('endpoint exc:', e)
    try:
        res = m.parse_images_with_gemini([])
        print('parse_images_with_gemini returned:', res)
    except Exception as e:
        print('parse_images_with_gemini exc:', e)
except Exception as e:
    print('import exc:', e)
