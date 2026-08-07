from dataclasses import dataclass


@dataclass
class RetryState:
    retries: int = 0
    max_retries: int = 2

    def can_retry(self) -> bool:
        return self.retries < self.max_retries

    def increase(self):
        self.retries += 1