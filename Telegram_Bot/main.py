import telegram

api_key = '1496208951:AAFkZoslJQifuCUDXCa4CSF-PoUDTNcKsaE'

bot = telegram.Bot(token=api_key)

chat_id = bot.get_updates()[-1].messege.chat_id

print(chat_id)



