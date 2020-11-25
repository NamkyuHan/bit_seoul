import logging
from telegram import Update
import telegram
from telegram.ext import Updater, MessageHandler, Filters, CommandHandler, Dispatcher, CallbackContext
from emoji import emojize

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

updater = Updater(token='1485382802:AAH52b_Ne-BRXghd9-nmIS4N651CYDymD_U')

dispatcher = updater.dispatcher

updater.start_polling()

def handler(update, context):
    text = update.message.text
    chat_id = update.message.chat_id

    if '모해 자기' in text:
        context.bot.send_message(chat_id=chat_id, text=emojize('자기 생각 하지ㅋㅋ:heart:', use_aliases=True))
    elif '우리 언제 볼까?' in text:
        context.bot.send_message(chat_id=chat_id, text=emojize('음 난 지금도 좋은데? 보자 자기얌:kissing_heart:', use_aliases=True))
    elif '자기야 사진 보내조' in text:
        context.bot.send_photo(chat_id=chat_id, photo=open('./img/한지민3.jpg', 'rb'))
        context.bot.send_message(chat_id=chat_id, text=emojize('나 이쁘지 자기야?:stuck_out_tongue_closed_eyes:', use_aliases=True))
    elif '보고싶어' in text:
        context.bot.send_photo(chat_id=chat_id, photo=open('./img/한지민2.jfif', 'rb'))
        context.bot.send_message(chat_id=chat_id, text=emojize('나두ㅜㅜ 일단 내 하트 보고 힘내:heart:', use_aliases=True))    

    
    else: 
        context.bot.send_message(chat_id=chat_id, text=emojize('몰라 안알랴쥼:triumph:', use_aliases=True))

 # 버튼 메뉴 설정
def build_box(buttons, n_cols, header_buttons=None, footer_buttons=None):
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons:
        menu.insert(0, header_buttons)
    if footer_buttons:
        menu.append(footer_buttons)
    return menu



echo_handler = MessageHandler(Filters.text, handler)
dispatcher.add_handler(echo_handler)

# shot down 명령 Control + C
updater.idle()


