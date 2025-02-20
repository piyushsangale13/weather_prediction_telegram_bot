from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CommandHandler, CallbackContext
import requests
import os

# Replace with your bot token
BOT_TOKEN = "7995166611:AAFIYNOpC7EmnMDlqP4_4kaDholYrAKO_rk"
FLASK_SERVER_URL = "http://127.0.0.1:5000/predict"  # Change to deployed server URL

# Function to handle incoming images
async def handle_image(update: Update, context: CallbackContext):
    file = await update.message.photo[-1].get_file()  # Get the highest resolution image
    file_path = "temp.jpg"
    await file.download_to_drive(file_path)  # Download the image

    # Send the image to the Flask server
    with open(file_path, "rb") as img:
        files = {"file": img}
        response = requests.post(FLASK_SERVER_URL, files=files)
    
    # Get the prediction result
    if response.status_code == 200:
        prediction = response.json()["prediction"]
        await update.message.reply_text(f"Predicted Weather: {prediction}")
    else:
        await update.message.reply_text("Error processing image.")

    os.remove(file_path)  # Clean up

# Start command handler
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text("Send me a weather image, and I'll predict the weather!")

# Set up the bot
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    # Add handlers
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_image))

    print("Bot is running...")
    app.run_polling()

if __name__ == "__main__":
    main()
