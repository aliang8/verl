import threading
import queue
from contextlib import contextmanager
import logging
from llm_sandbox import SandboxSession
from llm_sandbox.exceptions import SandboxTimeoutError
from typing import List

logger = logging.getLogger(__name__)


class SafeResourceManagedExecutor:
    """Combined robust code executor with resource management that reuses sessions"""

    def __init__(self, max_concurrent=10):
        self.semaphore = threading.Semaphore(max_concurrent)
        self.execution_queue = queue.Queue()
        self.session = self._init_sandbox()
        self.installed_libs = set()

    def _init_sandbox(self):
        """Initialize SandboxSession with proper error handling."""
        session = SandboxSession(
            lang="python",
            execution_timeout=10,
            verbose=False,
            runtime_configs={"cpu_count": 50, "mem_limit": "4096m"},
        )
        session.open()
        print("SandboxSession initialized and opened successfully")
        return session

    def close(self):
        """Clean up resources."""
        if self.session is not None:
            self.session.close()
            print("SandboxSession closed successfully")

    @contextmanager
    def acquire_resources(self):
        """Acquire execution resources"""
        self.semaphore.acquire()
        try:
            yield
        finally:
            self.semaphore.release()

    def execute_safely(self, code: str, libraries: List[str], **kwargs):
        """Execute code safely with comprehensive error handling and resource management"""

        for lib in libraries:
            if lib not in self.installed_libs:
                self.session.install([lib])
                self.installed_libs.add(lib)

        # Pre-execution validation
        if not code or not code.strip():
            return {
                "error": "Empty code provided",
                "error_type": "empty_code",
                "exit_code": 1,
                "stdout": "",
                "stderr": "Empty code provided",
            }

        with self.acquire_resources():
            assert self.session is not None

            # Security check if available
            if hasattr(self.session, "is_safe"):
                try:
                    is_safe, violations = self.session.is_safe(code)
                    if not is_safe:
                        return {
                            "error": "Security violation",
                            "error_type": "security_violation",
                            "violations": [
                                (
                                    v.description
                                    if hasattr(v, "description")
                                    else str(v)
                                )
                                for v in violations
                            ],
                            "exit_code": 1,
                            "stdout": "",
                            "stderr": "Security violation detected",
                        }
                except Exception as e:
                    logger.warning(
                        f"Security check failed: {e}, proceeding with execution"
                    )

            # Execute using the reused session
            try:
                # random hotfixes here
                code = code.replace('"\n"', '"\\n"')
                code = code.replace('\n"', '\\n"')
                result = self.session.run(code, libraries=libraries)
            except SandboxTimeoutError as e:
                return {
                    "error": "Execution timed out",
                    "error_type": "timeout_error",
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": str(e),
                }
            except Exception as e:
                return {
                    "error": "Execution failed",
                    "error_type": "execution_failed",
                    "exit_code": 1,
                    "stdout": "",
                    "stderr": str(e),
                }

            # Post-execution validation
            if result.exit_code != 0:
                error_type = "execution_failed"
                # Categorize error types based on stderr content
                stderr_lower = result.stderr.lower()
                if "memoryerror" in stderr_lower or "memory" in stderr_lower:
                    error_type = "memory_error"
                elif "timeout" in stderr_lower:
                    error_type = "timeout_error"
                elif "syntaxerror" in stderr_lower:
                    # import ipdb; ipdb.set_trace()
                    error_type = "syntax_error"
                elif (
                    "importerror" in stderr_lower
                    or "modulenotfounderror" in stderr_lower
                ):
                    error_type = "import_error"
                elif "assertionerror" in stderr_lower:
                    error_type = "assertion_error"

                return {
                    "error": "Execution failed",
                    "error_type": error_type,
                    "stderr": result.stderr,
                    "stdout": result.stdout,
                    "exit_code": result.exit_code,
                }

            return {
                "success": True,
                "output": result.stdout,
                "stderr": result.stderr,
                "stdout": result.stdout,
                "exit_code": result.exit_code,
            }