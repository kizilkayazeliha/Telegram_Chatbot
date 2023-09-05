
from telegram.ext import MessageHandler, CommandHandler, Filters, Updater
import nlp



def chat(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text=nlp.chat(update.message.text))


def start(update, context):
    context.bot.send_message(chat_id=update.effective_chat.id, text="Merhaba")


def main():
    updater = Updater(token='...', use_context=True)
    dispatcher = updater.dispatcher

    chat_handler = MessageHandler(Filters.text, chat)
    dispatcher.add_handler(chat_handler)

    start_handler = CommandHandler('start', start)
    dispatcher.add_handler(start_handler)

    updater.start_polling()
    updater.idle()


if __name__ == "__main__":
    main()