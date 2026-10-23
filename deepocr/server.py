from deeplabel_backend import serve

from deepocr.model import DeepOCRBackend

if __name__ == '__main__':
    serve(DeepOCRBackend())
