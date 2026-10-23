from deeplabel_backend import serve

from kraken_lines.model import KrakenBackend

if __name__ == '__main__':
    serve(KrakenBackend())
