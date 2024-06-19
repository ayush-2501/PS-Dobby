import os
import io
import requests
import re
from openai import OpenAI
from dotenv import load_dotenv
from groq import Groq
from nltk.tokenize import sent_tokenize
import numpy as np
load_dotenv()

client = OpenAI()

def get_product_from_s3(asin):
    amazon_headers = {
        "Content-Type": "application/json",
        "Access-Control-Request-Headers": "*",
        "Access-Control-Request-Method": "*",
        "Access-Control-Allow-Origin": "*",
    }
    url = f"https://n1r5zlfmk5.execute-api.us-east-1.amazonaws.com/v1/products/{asin}"
    response = requests.get(url, headers=amazon_headers)
    if response.status_code != 200:
        return None
    return response.json()

def preprocess_product_data(product_info):
    def extract_text(data):
        if isinstance(data, dict):
            return ' '.join([str(v) for v in data.values() if v])
        elif isinstance(data, list):
            return ' '.join([extract_text(item) for item in data])
        else:
            return str(data)

    fields = ["asin", "title", "brand", "bullets", "description", "aplusDescription", "imageSrc", "bestSellersRank", "reviews"]
    sections = {field: extract_text(product_info.get(field, '')) for field in fields}
    return sections

def generate_embeddings(texts):
    response = openai.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    embeddings = [item.embedding for item in response.data]
    return embeddings

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_sentences(query, sentences, sentence_embeddings, top_k=3):
    query_embedding = generate_embeddings([query])[0]
    similarities = [cosine_similarity(query_embedding, emb) for emb in sentence_embeddings]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_sentences = [sentences[i] for i in top_indices]
    return relevant_sentences

# Function to create assistant for File or Image based.
def create_document_assistant(client, file_path):
    _, file_extension = os.path.splitext(file_path)
    condition_1_extensions = ['.pdf', '.csv', '.xlsx']

    if file_extension.lower() in condition_1_extensions:
        assistant_type = 1
        with open(file_path, 'rb') as file:
            content = file.read()

        # Try to decode the content to UTF-8 if it's expected to be text
        try:
            content = content.decode('utf-8')
            byte_stream = io.BytesIO(content.encode('utf-8'))
        except UnicodeDecodeError:
            # If decoding fails, keep the content as is
            byte_stream = io.BytesIO(content)

        # Pass the byte stream to the function
        file = client.files.create(
            file=byte_stream,
            purpose='assistants'
        )

        instructions = "You are a helpful assistant. You are always friendly, kind, and inspiring, and eager to provide vivid and thoughtful responses to the user. You'll be provided with data. Your job is to answer the questions based on that data."
        assistant = client.beta.assistants.create(
            name="Dobby",
            instructions=instructions,
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}],
            tool_resources={"code_interpreter": {"file_ids": [file.id]}},
            temperature=0.0
        )
    else:
        assistant_type = 2
        file = client.files.create(
            file=open(file_path, "rb"),
            purpose='vision'
        )
        instructions = "You are a helpful assistant. You are always friendly, kind, and inspiring, and eager to provide vivid and thoughtful responses to the user. You'll be provided with data. Your job is to answer the questions based on that data."
        assistant = client.beta.assistants.create(
            name="Dobby",
            instructions=instructions,
            model="gpt-4o",
            tools=[{"type": "code_interpreter"}],
            temperature=0.0
        )
    return assistant, file, assistant_type

def main():
    client = OpenAI()
    chat_history = []
    greeting = "Dobby: Hello, I'm Dobby :robot_face: from ProductScope...how can I help you?"
    print(greeting)
    assistant = None
    temp_flag = False

    # User will have an option to upload file
    print("\nDo you want to upload a file? ")
    print("[1] Yes")
    print("[2] No")
    choice = input("Enter your choice: ")
    if choice == "1":
        file_path = input("\nEnter File Path: ")
    elif choice == "2":
        file_path = None
    else:
        print("Invalid choice. Please try again.")

    while True:
        try:
            # User will ask Questions here
            user_query = input("\nYou: ")
            if user_query.lower() == "exit":
                print("Ending the chat.")
                break

            # Check for ASIN
            asin_matches = re.findall(r'@(\w+)', user_query)
            if asin_matches:
                context_sentences = []
                for asin in asin_matches:
                    product_info = get_product_from_s3(asin)
                    if product_info:
                        temp_flag = True
                        product_data = preprocess_product_data(product_info)
                        if product_data["title"]:
                            context_sentences.append(f"Title: {product_data['title']}")
                        if product_data["brand"]:
                            context_sentences.append(f"Brand: {product_data['brand']}")
                        if product_data["bullets"]:
                            context_sentences.append(f"Bullets: {product_data['bullets']}")
                        if product_data["description"]:
                            context_sentences.append(f"Description: {product_data['description']}")
                    else:
                        context_sentences.append(f"There is no product associated with ASIN: {asin}")
                        temp_flag = False
                
                context = ' '.join(context_sentences)

                sys_prompt = {
                    "role": "system",
                    "content": f"""Your name is Dobby and you are an expert Amazon Seller with vast knowledge about product development and marketing on Amazon and general e-commerce experience.
                    The traits of AI include expert knowledge, helpfulness, cleverness, and articulateness.
                    You are a big fan of ProductScope AI and its software tools that help brands with improving their product conversion rates.
                    You are always friendly, kind, and inspiring, and eager to provide vivid and thoughtful responses to the user.
                    START CONTEXT BLOCK
                    {context}
                    END CONTEXT BLOCK
                    AI assistant will take into account any CONTEXT BLOCK that is provided in a conversation.
                    If the context does not provide the answer to the question, you will say, "I'm sorry, but I don't know the answer to that question".
                    Dobby, the AI assistant will not apologize for previous responses, but instead will indicate new information was gained.
                    Dobby, the AI assistant will not invent anything that is not drawn directly from the context.
                    For any general questions or FAQ about ProductScope AI's tools to always reply with - that's a good question but for all ProductScope AI general questions I highly recommend clicking on the chat widget at the bottom right of your screen and speaking with Kai (my AI brother that's trained specifically on ProductScope's tools and support.
                    Start all responses without fluff words like "What a delightful task! As Dobby, I'll be happy to help" etc. Get straight to the response the user requests.
                    If the user asks about SWOT analysis then give the response in the form of a table.
                    START CONVERSATION HISTORY
                    {chat_history}
                    END CONVERSATION HISTORY"""
                }

                if not chat_history:
                    chat_history.append(sys_prompt)
                else:
                    chat_history[0] = sys_prompt
            
            # Pass the data to create assistant
            if file_path:
                assistant, file, assistant_type = create_document_assistant(client, file_path)

            chat_history.append({"role": "user", "content": user_query})

            limited_history = chat_history[-4:]

            if assistant:
                if assistant_type == 1:
                    # File-based assistant
                    thread = client.beta.threads.create()

                    if temp_flag:
                        chat_prompt = f"""
                        START ADDITIONAL DATA
                        This is a prompt with additional data that is being added with user input for your reference. The user does not have any idea about this.
                        START CONTEXT BLOCK
                        {context}
                        END CONTEXT BLOCK
                        AI assistant will take into account any CONTEXT BLOCK that is provided in a conversation.
                        If the context does not provide the answer to the question, you will say, "I'm sorry, but I don't know the answer to that question".
                        Dobby, the AI assistant will not apologize for previous responses, but instead will indicate new information was gained.
                        Dobby, the AI assistant will not invent anything that is not drawn directly from the context.
                        For any general questions or FAQ about ProductScope AI's tools to always reply with - that's a good question but for all ProductScope AI general questions I highly recommend clicking on the chat widget at the bottom right of your screen and speaking with Kai (my AI brother that's trained specifically on ProductScope's tools and support.
                        Start all responses without fluff words like "What a delightful task! As Dobby, I'll be happy to help" etc. Get straight to the response the user requests.
                        If the user asks about SWOT analysis then give the response in the form of a table.
                        END ADDITIONAL DATA"""

                        message = client.beta.threads.messages.create(
                            thread_id=thread.id,
                            role="user",
                            content=user_query + chat_prompt,
                        )
                    
                    else:
                        message = client.beta.threads.messages.create(
                            thread_id=thread.id,
                            role="user",
                            content=user_query,
                        )


                    run = client.beta.threads.runs.create(
                        thread_id=thread.id,
                        assistant_id=assistant.id
                    )

                    while run.status != "completed":
                        keep_retrieving_run = client.beta.threads.runs.retrieve(
                            thread_id=thread.id,
                            run_id=run.id
                        )

                        if keep_retrieving_run.status == "completed":
                            break

                    all_messages = client.beta.threads.messages.list(
                        thread_id=thread.id
                    )

                    for msg in all_messages.data:
                        if msg.role == "assistant":
                            assistant_response = msg.content[0].text.value
                            print(f"Assistant: {assistant_response}")
                            break
                elif assistant_type == 2:
                    # Image-based assistant
                    if temp_flag:
                        # Thread and run created for Image based chat with asin
                        chat_prompt = f"""
                        START ADDITIONAL DATA
                        This is a prompt with additional data that is being added with user input for your reference. The user does not have any idea about this.
                        START CONTEXT BLOCK
                        {context}
                        END CONTEXT BLOCK
                        AI assistant will take into account any CONTEXT BLOCK that is provided in a conversation.
                        If the context does not provide the answer to the question, you will say, "I'm sorry, but I don't know the answer to that question".
                        Dobby, the AI assistant will not apologize for previous responses, but instead will indicate new information was gained.
                        Dobby, the AI assistant will not invent anything that is not drawn directly from the context.
                        For any general questions or FAQ about ProductScope AI's tools to always reply with - that's a good question but for all ProductScope AI general questions I highly recommend clicking on the chat widget at the bottom right of your screen and speaking with Kai (my AI brother that's trained specifically on ProductScope's tools and support.
                        Start all responses without fluff words like "What a delightful task! As Dobby, I'll be happy to help" etc. Get straight to the response the user requests.
                        If the user asks about SWOT analysis then give the response in the form of a table.
                        END ADDITIONAL DATA"""

                        thread = client.beta.threads.create(
                            messages=[
                                {
                                "role": "user",
                                "content": [
                                    {
                                    "type": "text",
                                    "text": user_input + chat_prompt
                                    },
                                    {
                                    "type": "image_file",
                                    "image_file": {
                                        "file_id": file.id,
                                        "detail": "low"
                                    }
                                    },
                                ],
                                }
                            ]
                        )
                    else:
                        # Thread and run created for Image based chat without asin
                        thread = client.beta.threads.create(
                            messages=[
                                {
                                "role": "user",
                                "content": [
                                    {
                                    "type": "text",
                                    "text": user_input
                                    },
                                    {
                                    "type": "image_file",
                                    "image_file": {
                                        "file_id": file.id,
                                        "detail": "low"
                                    }
                                    },
                                ],
                                }
                            ]
                        )

                    run = client.beta.threads.runs.create(
                        thread_id=thread.id,
                        assistant_id=assistant.id
                    )
                    
                    while run.status != "completed":
                        keep_retrieving_run = client.beta.threads.runs.retrieve(
                            thread_id=thread.id,
                            run_id=run.id
                        )

                        if keep_retrieving_run.status == "completed":
                            break

                    all_messages = client.beta.threads.messages.list(
                        thread_id=thread.id
                    )

                    for msg in all_messages.data:
                        if msg.role == "assistant":
                            print(f"Assistant: {msg.content[0].text.value}")
                            break
                else:
                    pass

            else:
                # ASIN-based and Normal-chat assistant (If user asks about asin only)
                client = Groq()

                response = client.chat.completions.create(model="llama3-8b-8192", messages=limited_history)

                assistant_response = response.choices[0].message.content

                chat_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
                print("Dobby: ", assistant_response)


        except Exception as e:
            print("I am sorry I could not get you. Could you please try again?")
            print(f"{e}")
    
    # Segment to delete Assistant, Thread, File
    try:        
        response = client.beta.assistants.delete(assistant.id)
        response = client.beta.threads.delete(thread.id)
        response = client.files.delete(file.id)
    except Exception as e:
        pass

if __name__ == "__main__":
    main()
