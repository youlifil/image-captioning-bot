import sys
from bot import run_bot
from local_test import run_local_test
import log

if __name__ == '__main__':
    log.init_logger()

    if len(sys.argv) == 2 and sys.argv[1] == 'localtest':
        run_local_test()
    else:
        run_bot()
