import telegram
import logging

api_key = '1485382802:AAH52b_Ne-BRXghd9-nmIS4N651CYDymD_U'

bot = telegram.Bot(token=api_key)

chat_id = bot.get_updates()[-1].message.chat_id
chat_id = 1477103062

# print(chat_id) 
#1477103062 피카츄
#1477103062 지민

#봇 정보 전달
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)
logger = logging.getLogger(__name__)

bot.sendMessage(chat_id=chat_id, text=' 안녕 남규야') 


