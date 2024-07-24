# BankBillValidatorBot


A Telegram bot for checking the authenticity of PDF files by comparing them with a provided original PDF. This bot performs various checks, including text similarity, image similarity, and metadata comparison.

## Features

- **Text Matching**: Compares the text in the PDF against an original document.
- **Image Similarity**: Uses SSIM and ORB methods to compare images.
- **Color Statistics**: Compares average colors, standard deviations, and color clusters.
- **Metadata Comparison**: Checks for discrepancies in metadata and fonts.
- **Inline Buttons**: Provides additional information through inline buttons.

## Prerequisites

- Python 3.8 or higher
- A Telegram bot token

## Installation

1. **Clone the repository**
   ```sh
   git clone https://github.com/yourusername/BankBillValidatorBot.git
   cd BankBillValidatorBot
   ```

2. **Create a virtual environment (optional but recommended)**
   ```sh
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required dependencies**
   ```sh
   pip install -r requirements.txt
   ```

4. **Set up your Telegram bot token and users eligible to use**
   Open the main.py file and replace "YOUR_TELEGRAM_BOT_TOKEN" with your actual bot token and users array
   
6. **Run the bot**
   ```sh
   python bot.py
   ```
   
