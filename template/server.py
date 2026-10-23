from deeplabel_backend import serve
from model import MyModel

if __name__ == '__main__':
    serve(MyModel())
