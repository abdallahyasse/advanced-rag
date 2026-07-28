import time

from starlette.middleware.base import BaseHTTPMiddleware

from app.logging.logger import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):

    async def dispatch(
        self,
        request,
        call_next,
    ):

        start = time.perf_counter()

        logger.info(
            f"Request {request.method} {request.url.path}"
        )

        response = await call_next(request)

        elapsed = time.perf_counter() - start

        logger.info(
            f"Response {response.status_code} ({elapsed:.3f}s)"
        )

        return response