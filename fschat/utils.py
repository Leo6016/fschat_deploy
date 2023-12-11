import logging
import logging.handlers
import sys
import os
from enum import IntEnum
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

handler = None
visited_loggers = set()

def pretty_print_semaphore(semaphore):
    """Print a semaphore in better format."""
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"


def build_logger(logger_name, logger_filename, logger_dir):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.basicConfig(level=logging.INFO, encoding="utf-8")
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # if LOGDIR is empty, then don't try output log to local file
    if logger_dir != "":
        os.makedirs(logger_dir, exist_ok=True)
        filename = os.path.join(logger_dir, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(
            filename, when="D", utc=True, encoding="utf-8"
        )
        handler.setFormatter(formatter)

        for l in [stdout_logger, stderr_logger, logger]:
            if l in visited_loggers:
                continue
            visited_loggers.add(l)
            l.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                encoded_message = line.encode("utf-8", "ignore").decode("utf-8")
                self.logger.log(self.log_level, encoded_message.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            encoded_message = self.linebuf.encode("utf-8", "ignore").decode("utf-8")
            self.logger.log(self.log_level, encoded_message.rstrip())
        self.linebuf = ""


class ErrorCode(IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006


SEQUENCE_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_position_embeddings",
    "max_seq_len",
    "model_max_length",
]

def get_context_length(config):
    """Get the context length of a model from a huggingface model config."""
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = config.rope_scaling["factor"]
    else:
        rope_scaling_factor = 1

    for key in SEQUENCE_LENGTH_KEYS:
        val = getattr(config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


class ModelRequest(BaseModel):
    model: str
    params: Optional[dict] = {}

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class ErrorResponse(BaseModel):
    object: str = "error"
    message: str
    code: int

class DeployCommand(BaseModel):
    deploy_method: str = Field(..., description="Enter 'hf' or 'vllm'")
    model_path: str
    host: str
    port: int
    model_names: List[str]
    gpu: List[str]
    additonal_params: Optional[dict]

class killCommand(BaseModel):
    host: str
    port: int

class IPRequest(BaseModel):
    host: str

class daemonInfo(BaseModel):
    daemon_name: str
    daemon_IP: str
    daemon_port: int
