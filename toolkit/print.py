import sys
import os
from toolkit.accelerator import get_accelerator


def print_acc(*args, **kwargs):
    if get_accelerator().is_local_main_process:
        try:
            # Under TPU multi-core spawn every worker is "local main" on its
            # own Accelerator; only ordinal 0 prints so logs don't repeat 8x.
            from toolkit.xla_utils import is_master_ordinal
            if not is_master_ordinal():
                return
        except Exception:
            pass
        print(*args, **kwargs)


class Logger:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # Make sure it's written immediately

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def isatty(self):
        return self.terminal.isatty()


def _is_log_process():
    if not get_accelerator().is_local_main_process:
        return False
    try:
        # Under TPU multi-core spawn every worker is "local main" on its own
        # Accelerator; only ordinal 0 owns the log file.
        from toolkit.xla_utils import is_master_ordinal
        return is_master_ordinal()
    except Exception:
        return True


def setup_log_to_file(filename):
    if not _is_log_process():
        return
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    # Capture the real streams before replacing them — wrapping the
    # already-replaced sys.stdout as the stderr Logger's "terminal" would
    # double-write every stderr message to the file. Both wrappers share a
    # single file handle.
    log_file = open(filename, 'a')
    sys.stdout = Logger(sys.stdout, log_file)
    sys.stderr = Logger(sys.stderr, log_file)
