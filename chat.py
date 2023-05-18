from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ChatVectorDBChain
from langchain.llms import OpenAI
from flask import Flask, request, jsonify

import slack
import multiprocessing

OPEN_AI_API_KEY="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"
SLACK_API_TOKEN="xoxb-2526947220-5102269875637-04EngCGKY6YlSt3Gf7LBdQeW"

pdf_path = "./paper.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load_and_split()
#print(pages[0].page_content)

embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API_KEY)
vectordb = Chroma.from_documents(pages, embeddings=embeddings, persist_directory = ".")
vectordb.persist()

def get_answer(query):
    open_ia_client = OpenAI(temperature=0.5, model_name="gpt-3.5-turbo", openai_api_key=OPEN_AI_API_KEY)
    pdf_qa = ChatVectorDBChain.from_llm(open_ia_client, vectordb, return_source_documents=True)
    result = pdf_qa({"question": query, "chat_history": ""})
    return result["answer"]


slack_client = slack.WebClient(token=SLACK_API_TOKEN)
# Manejar eventos de mensaje
@slack.RTMClient.run_on(event='message')
def handle_message(**payload):
    data = payload['data']
    if 'text' in data:
        message = data['text']
        print("Message:")
        print(message)
        channel_id = data['channel']
        user_id = data['user']
        answer = get_answer(message)
        print("Answer:")
        print(answer)
        # Realizar acciones basadas en el mensaje
        if message.lower() == 'hola':
            send_message(channel_id, answer)
        else:
            send_message(channel_id, answer)

def send_message(channel, text):
    slack_client.chat_postMessage(channel=channel, text=text)

# Iniciar el cliente de Slack en tiempo real
def start_slack_client():
    rtm_client = slack.RTMClient(token=SLACK_API_TOKEN)
    rtm_client.start()


def start_api_client():
    app = Flask(__name__)
    @app.route('/api/chat', methods=['POST'])
    def chat():
        data = request.get_json()
        origin = data.get('origin')
        destination = data.get('destination')
        adults = data.get('pasengers.adults')
        children = data.get('pasengers.children')
        places_promt = f'Que lugares hay disponibles para que {children} niños y {adults} adultos visiten {destination}'      
        places_answer = get_answer(places_promt)

        activities_promt = f'Que actividades hay entre en {destination}'      
        activities_answer = get_answer(activities_promt)

        wheather_promt = f'Que tipo de clima se ofrece en {destination}, responde sólo con una de las opciones de cálido, templado, lluvioso o frio'
        wheather_answer = get_answer(wheather_promt)

        beach_promt = f'Hay playas en {destination}, responde sólo con sí o no'
        beach_answer = get_answer(beach_promt)

        return jsonify({'places': places_answer, 'activities': activities_answer, 'wheather': wheather_answer, 'has_beach': beach_answer})
    app.run(port=3008)

if __name__ == '__main__':
    start_api_client()
    # p1 = multiprocessing.Process(target=start_slack_client)
    # p2 = multiprocessing.Process(target=start_api_client)

    # # Iniciar los procesos
    # p1.start()
    # p2.start()

    # # Esperar a que los procesos terminen
    # p1.join()
    # p2.join()
