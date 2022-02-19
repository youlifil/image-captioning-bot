import telebot
from auth import bot_token
# from local_test import run_local_test

import PIL
import urllib
import numpy as np

import image_captions as imcap


def message_image(bot, message):
    image_id = message.photo[-1].file_id
    file_path = bot.get_file(image_id).file_path
    image_url = "https://api.telegram.org/file/bot{0}/{1}".format(bot_token, file_path)
    image = np.array(PIL.Image.open(urllib.request.urlopen(image_url)))
    return image


def run_bot():
    bot = telebot.TeleBot(bot_token)

    @bot.message_handler(commands=["start"])
    def start_message(message):
        bot.send_message(message.chat.id, "Привет. Шлите картинки, а я скажу, что на них видно.")
    
    @bot.message_handler(content_types=['photo'])
    def generate_image_captions(message):
        print("picture from {}".format(message.from_user.username))
        bot.send_message(message.chat.id, "Подождите немножко... (или множко)")

        image = message_image(bot, message)
        captions = imcap.generate_captions(image)

        bot.send_message(message.chat.id, '\n'.join(['• ' + cap for cap in captions]))
        print('\n'.join(captions), '\n')

    bot.polling(non_stop=True)
    

if __name__ == '__main__':
    imcap.init_model()
    run_bot()
    # run_local_test()