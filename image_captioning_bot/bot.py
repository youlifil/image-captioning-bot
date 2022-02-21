import telebot
from image_captioning_bot.auth import bot_token
import image_captioning_bot.log as log
from tg_tqdm import tg_tqdm

import PIL
import urllib
import numpy as np

import image_captioning_bot.model as model

HELLO_STR = """
Привет, я бот, я делаю Image Captioning.
(итоговый проект для DLS семестр 2)

Мой код живёт здесь: https://github.com/youlifil/image-captioning-bot
"""

def _progress_bar(step, total, chat_id):
    if step == 0:
        _progress_bar.bar = tg_tqdm(range(total), bot_token, chat_id)
    _progress_bar.bar.update(1)


def run_bot():
    logger = log.logger()
    model.init_model()

    bot = telebot.TeleBot(bot_token)

    def image_from_message(bot, message):
        image_id = message.photo[-1].file_id
        file_path = bot.get_file(image_id).file_path
        image_url = "https://api.telegram.org/file/bot{0}/{1}".format(bot_token, file_path)
        image = np.array(PIL.Image.open(urllib.request.urlopen(image_url)))
        return image

    @bot.message_handler(commands=["start"])
    def start_message(message):
        bot.send_message(message.chat.id, HELLO_STR)
        bot.send_message(message.chat.id, "Шлите сюда картинки, а я скажу, что на них видно.")
    
    @bot.message_handler(content_types=['photo'])
    def receive_image_generate_captions(message):
        logger.info("picture from {}".format(message.from_user.username))
        bot.send_message(message.chat.id, "Подождите немножко... (или множко)")

        image = image_from_message(bot, message)
        captions = model.generate_captions(
                        image, 
                        step_callback=lambda step, total: _progress_bar(step, total, message.chat.id))

        bot.send_message(message.chat.id, '\n'.join(['• ' + cap for cap in captions]))
        logger.info('Generated captions are:\n' + '\n'.join(captions) + '\n')

    bot.polling(non_stop=True)
    
