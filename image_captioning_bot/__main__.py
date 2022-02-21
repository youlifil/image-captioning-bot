import sys
from image_captioning_bot.bot import run_bot
from image_captioning_bot.local_test import run_local_test
import image_captioning_bot.log as log

if __name__ == '__main__':
    log.init_logger()

    if len(sys.argv) == 2 and sys.argv[1] == 'localtest':
        run_local_test()
    else:
        run_bot()
