#import modules
from telegram.ext import (Updater, CommandHandler, MessageHandler, Filters,)
import logging

# Enable logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
#My bot token from BotFather
token = '1496208951:AAFkZoslJQifuCUDXCa4CSF-PoUDTNcKsaE'

# define command handlers
def start(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="봇 작동합니다.")

# 정해진 커맨드가 아닌 다른 명령을 받았을 때 출력할 메시지
def unknown(bot, update):
    bot.send_message(chat_id=update.message.chat_id, text="죄송하지만 그 명령어를 이해할 수 없습니다.")

# main문을 정의하고
def main():
    # Create Updater object and attach dispatcher to it
    updater = Updater(token)
    dp = updater.dispatcher
    print("Bot started")

    # Start the bot
    updater.start_polling()
    dp.add_handler(CommandHandler('start', start))
    dp.add_handler(MessageHandler(Filters.command, unknown))

    # Run the bot until you press Ctrl-C
    updater.idle()
    updater.stop()

if __name__ == '__main__':
    main()