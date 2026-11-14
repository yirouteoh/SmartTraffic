
import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

token = os.getenv("HF_TOKEN")

if not token:
    print("ERROR: HF_TOKEN was not found in .env")

else:
    print("Token detected successfully.")

    try:
        api = HfApi(token=token)
        user = api.whoami()

        print("Hugging Face authentication successful!")
        print("Authenticated account:", user.get("name"))

    except Exception as error:
        print("Authentication failed:", error)
