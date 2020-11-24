import requests
import json
import datetime
import time 
import logging
import telegram #pip install python-telegram-bot
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackQueryHandler, CallbackContext
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

#봇 정보 전달
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)

#Botfather token
token = "1496208951:AAFkZoslJQifuCUDXCa4CSF-PoUDTNcKsaE"

# 첫 시작
def start(update, context):
   #사용자 name 
    print(update.message.chat.username)
    t = ("안녕 %s!, 나는 맥주 추천봇이야!" + "\n" + "아직 서비스 준비 중이지만.." + "\n" + "내가 누군지 궁금하면 /who 를 눌러줘.") % update.message.chat.first_name
    context.bot.sendMessage(chat_id=update.message.chat_id, text=t)

# 사용자가 반복하기 예상이외에 다른 답변을 했을 때, 처음 멘트로 제공 
def echo(update, context):
    print(update.message.chat.username)
    t =("안녕 %s, 나는 맥주 추천봇이야!" + "\n" + " 아직 서비스 준비 중이지만.."+"\n" +"내가 누군지 궁금하면 /who 를 눌러줘") % update.message.chat.first_name
    context.bot.send_message(chat_id=update.message.chat_id, text=t)

# /who command
def who(update, context):
    t = "나는 맥주 데이터를 모아 분석해서 오늘 너의 기분에 따라 맥주 추천해주는 똑똑한 너의 술친구야!!" + "\n" + "이제 날 알겠지?" + "\n" + "/test 를 누르면 내가 맥주 하나 추천해줄게"
    context.bot.send_message(chat_id=update.message.chat_id, text=t)

    # 버튼 메뉴 설정
def build_box(buttons, n_cols, header_buttons=None, footer_buttons=None):
    menu = [buttons[i:i + n_cols] for i in range(0, len(buttons), n_cols)]
    if header_buttons:
        menu.insert(0, header_buttons)
    if footer_buttons:
        menu.append(footer_buttons)
    return menu
 
# /test commend
def test(update, context):
    t = "오늘은 날씨가 추우니깐 이런 날은 진한~ 흑맥주가 잘 어울려!"
    context.bot.sendMessage(chat_id=update.message.chat_id, text=t)
    time.sleep(0.3)
    #image 값 입력
    context.bot.send_photo(chat_id=update.message.chat_id, photo='https://www.gwine.com/images/labels/guinness-guinness-extra-stout.gif')
    time.sleep(0.3)
    t1 = "기네스 어때?!"
    context.bot.send_message(chat_id=update.message.chat_id, text=t1)
    time.sleep(0.3)
    #키보드에 대답 넣기
    show_list = []
    show_list.append(InlineKeyboardButton("좋아", callback_data="좋아"))
    show_list.append(InlineKeyboardButton("별로야", callback_data="별로야"))
    show_markup = InlineKeyboardMarkup(build_box(show_list, len(show_list) - 1)) # make markup
    update.message.reply_text("내 추천이 어떤지 알려 줄래?", reply_markup=show_markup)

# callback
def callback_get(update, context):
    print("callback")
    if update.callback_query.data == "좋아":
        context.bot.edit_message_text(text="진짜? 내 추천 좋지?! 오늘 술 잘 마시고 지나친 음주는 몸에 안 좋은 거 알지?!" + "\n" + "다음에 또 놀러와!",
                          chat_id=update.callback_query.message.chat_id,
                          message_id=update.callback_query.message.message_id)

    if update.callback_query.data == '별로야' :
        context.bot.edit_message_text(text= "솔직한 의견 고마워" + "\n" + "다음에 또 놀러와!",
                        chat_id = update.callback_query.message.chat_id,
                        message_id = update.callback_query.message.message_id)

# error 처리
def error(update, context, error):
    logger.warning('Update "%s" caused error "%s"', update, error)

# command & function 활성화 하기
def main():
    updater = Updater(token=token)
    dp = updater.dispatcher
    dp.add_handler(CommandHandler('start', start))
    #echo function은 Messagehandler
    dp.add_handler(MessageHandler(Filters.text, echo))
    #dp.add_handler(CommandHandler('beer', beer, pass_args=True))
    dp.add_handler(CommandHandler('who', who))
    dp.add_handler(CommandHandler('test', test))
    dp.add_handler(CallbackQueryHandler(callback_get))

    # log all errors
    dp.add_error_handler(error)
    # polling시작, 걸리는 시간 최대치 정해줌 너무 낮은 경우는 poll이 제대로 작동이 안됨
    # clean=true 기존의 텔레그램 서버에 저장되어있던 업데이트 사항 지우기
    updater.start_polling(timeout=3)
    # idle은 updater가 종료되지 않고 계속 실행
    updater.idle()

if __name__ == '__main__':
    main()                        