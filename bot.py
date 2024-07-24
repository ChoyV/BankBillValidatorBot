import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackContext, filters, CallbackQueryHandler
import logging
from main import check_authenticity  # Ensure 'main' module is correctly imported and available

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
ORIGINAL_PDF_PATH = 'Original.pdf'
FAKE_PDF_PATH = 'received_fake.pdf'

# Define your donation address
DONATION_ADDRESS = "ADDRESS_OF_WALLET"

# Define allowed users by user ID or username
ALLOWED_USERS = ['user1', 'user2']

# Define description text for parameters
DESCRIPTION_TEXT = (
    "*Description of Verification Parameters:*\n\n"
    "1. **Text Match Ratio**: This parameter measures the degree of text similarity between the original and the checked document. A value close to 1 indicates high similarity.\n\n"
    "2. **Image Similarity (SSIM)**: Structural Similarity Index for images. A value close to 1 indicates high similarity between images.\n\n"
    "3. **Image Similarity (ORB)**: This parameter measures image similarity using the ORB (Oriented FAST and Rotated BRIEF) method. A value close to 1 indicates high similarity.\n\n"
    "4. **Average Color Difference**: This parameter measures the difference in average colors between two images. Smaller values indicate greater similarity.\n\n"
    "5. **Difference in Mean Values and Standard Deviations**: These parameters measure the difference in mean values and standard deviations of color channels in images. Smaller values indicate greater similarity.\n\n"
    "6. **Difference in Color Clusters**: This parameter measures the difference in dominant colors of images using K-Means clustering. Smaller values indicate greater similarity."
)

def is_user_allowed(update: Update) -> bool:
    user = update.message.from_user
    return str(user.id) in ALLOWED_USERS or user.username in ALLOWED_USERS

async def start(update: Update, context: CallbackContext) -> None:
    if not is_user_allowed(update):
        await update.message.reply_text('Sorry, you do not have access to this bot.')
        return
    await update.message.reply_text('Hello! Send me a PDF file for verification. I will check its authenticity.')

async def handle_document(update: Update, context: CallbackContext) -> None:
    if not is_user_allowed(update):
        await update.message.reply_text('Sorry, you do not have access to this bot.')
        return

    file = await update.message.document.get_file()
    file_path = FAKE_PDF_PATH

    # Check if the document is a PDF
    if update.message.document.mime_type != 'application/pdf':
        await update.message.reply_text('Error: This file is not a PDF. Please send a PDF file.')
        return

    try:
        await file.download_to_drive(file_path)  # Use the correct async method
        await update.message.reply_text('Received the document. Checking...')

        # Check authenticity of the uploaded PDF against the original
        result = check_authenticity(ORIGINAL_PDF_PATH, file_path)
        
        # Append the donation address as plain text
        result += f'\n\n💰 Donation address: {DONATION_ADDRESS}'
        
        # Create a button to get a description of the parameters
        keyboard = [[InlineKeyboardButton("Description of Parameters", callback_data='description')]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(result, reply_markup=reply_markup)
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        await update.message.reply_text(f'Error: {e}')
    
    finally:
        # Clean up by removing the file
        if os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Removed file: {file_path}")

async def button_handler(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    await query.answer()  # Acknowledge the callback query
    if query.data == 'description':
        await query.message.reply_text(DESCRIPTION_TEXT, parse_mode='Markdown')

def main() -> None:
    # Initialize the Application with your bot's token
    application = Application.builder().token("YOUR_BOT_TOKEN").build()

    # Register handlers
    application.add_handler(CommandHandler('start', start))
    application.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    application.add_handler(CallbackQueryHandler(button_handler))

    # Start polling
    application.run_polling()

if __name__ == '__main__':
    main()
