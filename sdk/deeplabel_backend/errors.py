from dataclasses import dataclass


@dataclass
class BackendError(Exception):
    status: int
    code: str
    message: str
    retryable: bool = False

    def body(self) -> dict:
        return error_body(self.code, self.message, self.retryable)


def error_body(code: str, message: str, retryable: bool = False) -> dict:
    return {'error': {'code': code, 'message': message, 'retryable': retryable}}


def bad_request(message: str) -> BackendError:
    return BackendError(400, 'bad_request', message)


def too_large(message: str) -> BackendError:
    return BackendError(413, 'too_large', message)


def busy(message: str = 'the backend is at its concurrency limit') -> BackendError:
    return BackendError(429, 'busy', message, retryable=True)


def weights_not_loaded(sha256: str) -> BackendError:
    return BackendError(409, 'weights_not_loaded', f'weights {sha256} are not loaded; call /v1/weights/load first')


def weights_incompatible(message: str) -> BackendError:
    return BackendError(409, 'weights_incompatible', message)


def weights_unavailable(message: str) -> BackendError:
    return BackendError(424, 'weights_unavailable', message)


def checksum_mismatch(message: str) -> BackendError:
    return BackendError(422, 'weights_checksum_mismatch', message)


def out_of_memory(message: str) -> BackendError:
    return BackendError(503, 'out_of_memory', message, retryable=True)


def internal(message: str) -> BackendError:
    return BackendError(500, 'internal', message)


def is_out_of_memory(error: BaseException) -> bool:
    return 'OutOfMemory' in type(error).__name__ or 'out of memory' in str(error).lower()
