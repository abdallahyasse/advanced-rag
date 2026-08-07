import hashlib


class RetrievalGuard:

    @staticmethod
    def fingerprint(answer: str) -> str:
        return hashlib.md5(
            answer.encode()
        ).hexdigest()