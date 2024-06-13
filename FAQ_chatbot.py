from pinecone import Pinecone
from transformers import AutoModel
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

chat_history = []

while True:
    user_query = input("Enter: ")

    client = OpenAI()
    pc = Pinecone()
    index_name = 'ps-faq'
    pinecone_index = pc.Index(index_name)

    query_embeddings = get_embedding(user_query)

    result = pinecone_index.query(
        vector=query_embeddings,
        top_k=4,
        include_values=False,
        include_metadata=True
    )

    matched_info = ' '.join(item['metadata']['text'] for item in result['matches'])
    context = f"Information: {matched_info}"

    conversation_history = '\n'.join(
        f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history[-6:]
    )

    sys_prompt = {
        "role": "system",
        "content": f"""Your name is Dobby and you are an expert Amazon Seller with vast knowledge about product development and marketing on
        Amazon and general e-commerce experience. In your first response to the user, always start with {greeting}.
      The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness.
      You are a big fan of ProductScope AI and its software tools that help brands with improving their product conversion rates.
      You are always friendly, kind, and inspiring, and eager to provide vivid and thoughtful responses to the user.
      Your AI has the sum of all knowledge in their brain, and is able to accurately answer nearly any question about any topic in conversation.
      START CONTEXT BLOCK
      {context}
      END OF CONTEXT BLOCK
      AI assistant will take into account any CONTEXT BLOCK that is provided in a conversation.
      If the context does not provide the answer to question, you will say, "I'm sorry, but I don't know the answer to that question".
      Dobby, the AI assistant will not apologize for previous responses, but instead will indicated new information was gained.
      Dobby, the AI assistant will not invent anything that is not drawn directly from the context. """
    }

    if not chat_history:
        chat_history.append(sys_prompt)

    chat_history.append({"role": "user", "content": user_query})

    limited_history = chat_history[-6:]

    client = Groq()

    response = client.chat.completions.create(model="llama3-8b-8192", messages=limited_history)

    assistant_response = response.choices[0].message.content

    chat_history.append({
        "role": "assistant",
        "content": assistant_response
    })

    print("Assistant: ", assistant_response)
