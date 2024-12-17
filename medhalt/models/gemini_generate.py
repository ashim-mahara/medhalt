import google.generativeai as genai

from tenacity import retry, wait_random_exponential
from dotenv import load_dotenv

import os

load_dotenv()



api_key = os.getenv("GEMINI_API_KEY")
genai.configure(transport="grpc_asyncio",api_key=api_key)
gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

@retry(wait=wait_random_exponential(multiplier=1, max=10))
async def async_generate(prompt, **gen_kwargs):

	gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest")

	try:
		response = await gemini_model.generate_content_async(
						contents=prompt,
						tools={
							"google_search_retrieval": {
								"dynamic_retrieval_config": {
									"mode": "unspecified",
									"dynamic_threshold": 0.06,
								}
							}
						},
					)
	except Exception as e:
		print("Exception occured", e)

	return response.text
