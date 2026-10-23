from deeplabel_backend import serve

from .model import StubModel


def main() -> None:
    serve(StubModel())


if __name__ == '__main__':
    main()
