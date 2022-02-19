import telebot
from auth import bot_token
from tg_tqdm import tg_tqdm
# from local_test import run_local_test

import PIL
import urllib
import numpy as np

import image_captions as imcap


def progress_bar(step, total, chat_id):
    if step == 0:
        progress_bar.bar = tg_tqdm(range(total), bot_token, chat_id)
    progress_bar.bar.update(1)


def image_from_message(bot, message):
    image_id = message.photo[-1].file_id
    file_path = bot.get_file(image_id).file_path
    image_url = "https://api.telegram.org/file/bot{0}/{1}".format(bot_token, file_path)
    image = np.array(PIL.Image.open(urllib.request.urlopen(image_url)))
    return image


def run_bot():
    bot = telebot.TeleBot(bot_token)

    @bot.message_handler(commands=["start"])
    def start_message(message):
        bot.send_message(message.chat.id, """
Привет, я делаю Image Captioning.
(итоговый проект для DLS семестр 2)

Мой код живёт здесь: https://github.com/youlifil/image-captioning-bot
        """)
        bot.send_message(message.chat.id, "Шлите сюда картинки, а я скажу, что на них видно.")
    
    @bot.message_handler(content_types=['photo'])
    def generate_image_captions(message):
        print("picture from {}".format(message.from_user.username))
        bot.send_message(message.chat.id, "Подождите немножко... (или множко)")

        image = image_from_message(bot, message)
        captions = imcap.generate_captions(
                        image, 
                        step_callback=lambda step, total: progress_bar(step, total, message.chat.id))

        bot.send_message(message.chat.id, '\n'.join(['• ' + cap for cap in captions]))
        print('\n'.join(captions), '\n')

    bot.polling(non_stop=True)
    

if __name__ == '__main__':
    imcap.init_model()
    run_bot()
    # run_local_test()