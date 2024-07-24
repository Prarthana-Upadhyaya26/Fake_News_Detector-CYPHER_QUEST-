#from langchain_openai import OpenAI
# model = genai.GenerativeModel(
#   model_name="gemini-1.5-flash",
#   generation_config=generation_config,
#   system_instruction="You are a experienced saleswoman in a beauty shop with all the knowledge of cosmetics. Recommend items and products that compliments the current selection in the cart, be friendly, cheerful and polite.",
# )
# response = llm.invoke(
#         "What are some of the pros and cons of Python as a programming language?"
#     )

import sys
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
GEMINI_API_KEY = "GEMINI_API_KEY"

generation_config = {
    "temperature": 0.2,   # Lower temperature for more conservative and precise outputs
    "top_p": 1.0,         # Set top_p to 1.0 for deterministic outputs based on probability
    "top_k": 0,           # Disable top_k sampling for strict adherence to probabilities
    "max_output_tokens": 8192,   # Maximum number of output tokens to generate
    "response_mime_type": "text/plain",  # Output format type
}



llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

quest="I want to get more information about this particular statement: {your_text}. give me relevant urls.atleast provide some known credible news platforms or government related websites such as.dont just say u cant provide .just search and give me the url as result. I want a short para around 200 words explaining the conetents from these sources about the statement."

# chat_session = llm.model.start_chat(
#   history=[
#     {
#       "role": "user",
#       "parts": [
#         "An example output is given, please generate the further output in such a format only. The Cauvery River, flowing through the heart of South India, is a vital lifeline for the region. Its waters sustain agriculture, power industries, and quench the thirst of numerous cities in both Karnataka and Tamil Nadu. However, the river's importance has also sparked a long-standing dispute between the two states over water sharing. This dispute, rooted in historical grievances and competing claims over water rights, has led to tensions and legal battles.The Cauvery Water Disputes Tribunal (CWDT), established in 1990, attempted to resolve the issue by allocating water shares to both states. However, the implementation of the tribunal's award has been fraught with challenges, leading to further conflict. The dispute highlights the complex interplay of water resources, economic development, and political tensions in India. Here are some relevant URLs for further exploration:Government Website: https://cwatercommission.gov.in/ (Central Water Commission, India)News Source: https://www.thehindu.com/news/national/karnataka/cauvery-water-dispute-a-timeline/article65741779.ece (The Hindu)News Source: https://economictimes.indiatimes.com/news/how-to/explained-what-is-the-cauvery-water-dispute-why-is-karnataka-not-giving-water-to-tamil-nadu/articleshow/103954029.cms?utm_source=contentofinterest&utm_medium=text&utm_campaign=cppst (Economic Times)",
#       ],
#     },])

x = PromptTemplate(input_variables=["your_text"], template = quest)

if len(sys.argv) != 2:
    print("Usage: python subprocess_script.py <input>")
    sys.exit(1)

input_value = sys.argv[1]
# print(f"Received input: {input_value}")
x=x.format(your_text=input_value)
response = llm.invoke(x)
print(response)


