# Quick Start Guide

Follow these steps to get your chatbot running locally:

## Step 1: Install Dependencies

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

## Step 2: Set Up Hugging Face Token

1. Create a Hugging Face account at https://huggingface.co
2. Go to your profile → Settings → Access Tokens
3. Create a new token (read access is enough)
4. Create a `.env` file in the project root:
   ```
   HF_TOKEN=your_token_here
   ```
   Replace `your_token_here` with your actual token.

5. Also login via CLI (optional but recommended):
   ```bash
   huggingface-cli login
   ```

## Step 3: Run the Chatbot

```bash
python app.py
```

The chatbot will start and you'll see output like:
```
Running on local URL:  http://127.0.0.1:7860
```

Open that URL in your browser to use the chatbot!

## Troubleshooting

- **Import errors**: Make sure you activated your virtual environment and installed requirements
- **HF_TOKEN errors**: Check that your `.env` file exists and has the correct token
- **503 errors**: The free Hugging Face API sometimes has rate limits. Wait a few seconds and try again
- **Model errors**: Make sure you're logged into Hugging Face and have access to the model

## Next Steps

Once it's working, you can:
- Customize the system prompt in `src/chat.py` → `format_prompt()` method
- Test different models in `config.py`
- Improve the chatbot's responses
- Deploy to Hugging Face Spaces (see README.md)
