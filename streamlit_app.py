import streamlit as st
import replicate
import os
import pandas as pd

from openai import OpenAI
import openai
import requests

import time


from retriever import BM25Retriever, CERetriever, RetrieverEnsemble

title = "📚💬 Heidelberg University Examination Regulation Chat Bot"

# App title
st.set_page_config(page_title=title)

# Model selection
models = ['Llama2-7B-chat', 'leo-hessian-7B-chat', "GPT-3.5"]
if 'selected_model' not in st.session_state:
    st.session_state['selected_model'] = models[0]

# Language selection
languages = ['en', 'de']
if 'selected_language' not in st.session_state:
    st.session_state['selected_language'] = languages[0]

# Index selection
indices = ['English POs','translation of German POs', 'German POs', 'translation of English POs']
if 'selected_index' not in st.session_state:
    st.session_state['selected_index'] = indices[0]

if "ensemble_retriever" not in st.session_state:
    st.session_state.ensemble_retriever = ""

# Context length selection
if 'selected_context_length' not in st.session_state:
    st.session_state['selected_context_length'] = 10

if 'REPLICATE_API_TOKEN' in st.secrets:
    st.session_state['replicate_api'] = st.secrets['REPLICATE_API_TOKEN']
elif "REPLICATE_API_TOKEN" in os.environ:
    st.session_state['replicate_api'] = os.environ['REPLICATE_API_TOKEN']

if "evidence" not in st.session_state.keys():
    st.session_state.evidence = []

if "cqu" not in st.session_state.keys():
    st.session_state.cqu = ""

if "history" not in st.session_state:
    # Create a dataframe with the columns: turn, question, answer, evidence, cqu
    st.session_state.history = pd.DataFrame(columns=['turn', 'question', 'answer', 'evidence', 'cqu', 'context_relevance', 'context_position', 'faithfullness', 'answer_relevance', 'reason_for_incorrectness'])
    st.session_state.history = st.session_state.history.astype({
        'turn': 'int',
        'question': 'string',
        'answer': 'string',
        'evidence': 'string',
        'cqu': 'string',
        'context_relevance': 'bool',
        'context_position': 'int',
        'faithfullness': 'bool',
        'answer_relevance': 'float',
        'reason_for_incorrectness': 'string'
    })

if 'view' not in st.session_state:
    st.session_state['view'] = 'Settings'

if st.session_state.view != "Evaluation":

    # Replicate Credentials
    with st.sidebar:
        st.title(title)
        st.write("This is a testbed for the chatbot created for Stephan Lenert's Masterthesis.")
        
        # Add two buttons for view selection


        col1, col2 = st.columns(2)
        with col1:
            if st.button('Settings'):
                st.session_state['view'] = 'Settings'
        with col2:
            if st.button('Evidence'):
                st.session_state['view'] = 'Evidence'

        if st.session_state['view'] == 'Settings':

            st.title("⚙️ Settings")
            # Settings view
            if 'REPLICATE_API_TOKEN' in st.secrets:
                st.session_state['replicate_api'] = st.secrets['REPLICATE_API_TOKEN']
            else:
                if 'replicate_api' not in st.session_state:
                    st.session_state['replicate_api'] = ''
                st.session_state['replicate_api'] = st.text_input('Enter Replicate API token:', value=st.session_state['replicate_api'], type='password')
                if not (st.session_state['replicate_api'].startswith('r8_') and len(st.session_state['replicate_api'])==40):
                    st.warning('Please enter your credentials!', icon='⚠️')
                else:
                    st.success('Proceed to entering your prompt message!', icon='👉')
            os.environ['REPLICATE_API_TOKEN'] = st.session_state['replicate_api']

            if 'LEO_API_URL' in st.secrets:
                st.session_state['leo_api_url'] = st.secrets['LEO_API_URL']
            else:
                if 'replicate_api' not in st.session_state:
                    if "LEO_API_URL" in os.environ:
                        st.session_state['leo_api_url'] = os.environ['LEO_API_URL']
                    else:
                        st.session_state['leo_api_url'] = ''
                        st.session_state['leo_api_url'] = st.text_input('Enter Leo Hessian API URL:', value=st.session_state['leo_api_url'])

            if "leo_api_url" in st.session_state and "replicate_api" in st.session_state:
                st.success('You are ready to go! All APIs are provided.', icon='👍')

            st.subheader('Models and parameters')

            st.session_state['selected_model'] = st.selectbox('Choose a model', models, index=models.index(st.session_state['selected_model']), placeholder='Select a model')

            if st.session_state["selected_model"] == "Llama2-7B-chat":
                available_languages = ["en"] 
            elif st.session_state["selected_model"] == "leo-hessian-7B-chat":
                available_languages = ["de"]
            else:
                available_languages = languages

            st.session_state['selected_language'] = st.selectbox('Choose a language', available_languages, index=0, placeholder='Select a language')
            # st.session_state['selected_language'] = st.selectbox('Choose a language', languages, index=languages.index(st.session_state["selected_language"]), placeholder='Select a language')

            if st.session_state['selected_language'] == 'de':
                available_indices = indices[2:]
            else:
                available_indices = indices[:2]

            st.session_state['selected_index'] = st.selectbox('Choose an index', available_indices, index=available_indices.index(st.session_state['selected_index']) if st.session_state['selected_index'] in available_indices else 0, placeholder='Select an index')

            st.session_state['selected_context_length'] = st.slider('Choose a context length', min_value=1, max_value=10, value=st.session_state['selected_context_length'], step=1)

            # Write out selected settings
            st.write('Selected model:', st.session_state['selected_model'])
            st.write('Selected language:', st.session_state['selected_language'])
            st.write('Selected index:', st.session_state['selected_index'])
            st.write('Selected context length:', st.session_state['selected_context_length'])

        elif st.session_state['view'] == 'Evidence':
            st.title("🔍 Evidence")
            # Display CQU 

            st.write("Contextualized Query:")
            st.write(st.session_state.cqu)

            # Convert evidence to a DataFrame
            evidence_df = pd.DataFrame(st.session_state.evidence)

            # Transpose the DataFrame
            transposed_evidence = evidence_df.transpose()

            # Display the transposed DataFrame
            st.table(transposed_evidence[:st.session_state.selected_context_length])


    system_messages = {
        'de': {
            "start": "👋 Hallo! Ich kann dir bei Fragen zu deiner Prüfungsordnung helfen. 📚\n\n⏳ Wenn oben rechts 'Running' angezeigt wird, lädt noch etwas im Hintergrund. Das kann bis zu 2 Minuten dauern. Bitte änder auch keine Einstellungen während die Retriever intialisiert werden.\n\n⚙️ Links kannst du bei 'Settings' die Sprache, den Index und die Kontextlänge ändern.\n\n🔍 Unter 'Evidence' findest du die gefundenen Passagen und die verwendete kontextualisierte Frage.",
        },
        'en': {
            "start": "👋 Hello! I can assist you with questions about your examination regulations. 📚\n\n⏳ If 'Running' is still displayed in the top right corner, something is still loading in the background. This can take up to 2 minutes. Please don't change any settings while the retriever is beeing initalized.\n\n⚙️ On the left, you can change the language, the index, and the context length under 'Settings'.\n\n🔍 Under 'Evidence', you will find the passages found and the contextualized question used.",
        }
    }

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": system_messages[st.session_state.selected_language]["start"]}]


    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.cqu=""
        st.session_state.messages = [{"role": "assistant", "content": system_messages[st.session_state.selected_language]["start"]}]
        st.session_state.evidence = []
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    def evaluate_chat():
        st.session_state.view = "Evaluation"
    st.sidebar.button('Evaluate Chat', on_click=evaluate_chat)

    if st.session_state.selected_index == "English POs":
        index_name = "IndexEnglishAll.json"
    elif st.session_state.selected_index == "German POs":
        index_name = "IndexGermanAll.json"
    elif st.session_state.selected_index == "translation of English POs":
        index_name = "translated_IndexEnglishAll.json"
    elif st.session_state.selected_index == "translation of German POs":
        index_name = "translated_IndexGermanAll.json"

    with st.spinner('Initalizing Retrievers...'):
        if st.session_state.ensemble_retriever == "":
            bm25_retriever = BM25Retriever(index_name=index_name, language=st.session_state.selected_language, k=1500)
            ce_retriever = CERetriever(index_name=index_name, language=st.session_state.selected_language, k=10)
            st.session_state.ensemble_retriever = RetrieverEnsemble([bm25_retriever, ce_retriever], index_name=index_name, language=st.session_state.selected_language)
        elif st.session_state.ensemble_retriever.index_name != index_name or st.session_state.ensemble_retriever.language != st.session_state.selected_language:
            bm25_retriever = BM25Retriever(index_name=index_name, language=st.session_state.selected_language, k=1500)
            ce_retriever = CERetriever(index_name=index_name, language=st.session_state.selected_language, k=10)
            st.session_state.ensemble_retriever = RetrieverEnsemble([bm25_retriever, ce_retriever], index_name=index_name, language=st.session_state.selected_language)

    def retrieve_evidence():
        evidence = st.session_state.ensemble_retriever.get_top_k_documents(st.session_state.cqu)
        st.session_state.evidence = evidence 
        return evidence

    def generate_cqu_response():
        prompts = {
            'de': {
                'GPT-3.5': {
                    'system_prompt': "Als Assistent sollst du die aktuelle Frage unter Berücksichtigung eines mehrstufigen Informationsgesprächs dekontextualisieren und umformulieren. Bitte gib die umformulierte Frage nach 'Rewrite:' an. Hier sind einige Beispiele: \n\n 1. Beispiel:\nBenutzer: Wer ist mein Lieblingssänger?\nAssistent: Justin Bieber\nBenutzer: Wann wurde er geboren?\n\nRewrite: Wann wurde Justin Bieber geboren?\n\n2. Beispiel:\nBenutzer: Aus welcher Stadt in Deutschland kommt der Fußballclub KSC?\nAssistent: Karlsruhe\nBenutzer: Wie viele Menschen leben dort?\n\nRewrite: Wie viele Menschen leben in Karlsruhe?\n\n3. Beispiel:\nBenutzer: Wie heißt das Buch, das ich lese?\nAssistent: Das Buch, das Sie lesen, ist '1984' von George Orwell.\nBenutzer: Wer ist die Hauptfigur?\n\nRewrite: Wer ist die Hauptfigur in '1984' von George Orwell?\n\n4. Beispiel:\nBenutzer: Wie heißt der Regisseur des Films, den ich gestern gesehen habe?\nAssistent: Der Film, den Sie gestern gesehen haben, 'Inception', wurde von Christopher Nolan inszeniert.\nBenutzer: Welche anderen Filme hat er inszeniert?\n\nRewrite: Welche anderen Filme hat Christopher Nolan, der Regisseur von 'Inception', inszeniert?\n\n5. Beispiel:\nBenutzer: Wie heißt die Programmiersprache, die ich lerne?\nAssistent: Sie lernen Python.\nBenutzer: Wer hat es erstellt?\nAssistent: Python wurde von Guido van Rossum erstellt.\nBenutzer: Wann wurde es erstellt?\n\nRewrite: Wann wurde Python, die Programmiersprache, die von Guido van Rossum erstellt wurde, erstellt?",
                    'string_dialogue': "'Benutzer' bezieht sich auf Aussagen des Informationsuchenden in der Konversation. 'Assistent' auf Aussagen des Assistenten. Bitte formuliere die letzte Frage des Benutzers im Kontext um.",
                    'user_prompt': 'Benutzer',
                    'assistant_prompt': 'Assistent'
                },
                'leo-hessian-7B-chat': {
                    'system_prompt': "<|im_start|>system\nAls Assistent sollst du die aktuelle Frage unter Berücksichtigung eines mehrstufigen Informationsgesprächs dekontextualisieren und umformulieren. Bitte gib die umformulierte Frage nach 'Rewrite:' an. Hier sind einige Beispiele: \n\n 1. Beispiel:\nBenutzer: Wer ist mein Lieblingssänger?\nAssistent: Justin Bieber\nBenutzer: Wann wurde er geboren?\n\nRewrite: Wann wurde Justin Bieber geboren?\n\n2. Beispiel:\nBenutzer: Aus welcher Stadt in Deutschland kommt der Fußballclub KSC?\nAssistent: Karlsruhe\nBenutzer: Wie viele Menschen leben dort?\n\nRewrite: Wie viele Menschen leben in Karlsruhe?\n\n3. Beispiel:\nBenutzer: Wie heißt das Buch, das ich lese?\nAssistent: Das Buch, das Sie lesen, ist '1984' von George Orwell.\nBenutzer: Wer ist die Hauptfigur?\n\nRewrite: Wer ist die Hauptfigur in '1984' von George Orwell?\n\n4. Beispiel:\nBenutzer: Wie heißt der Regisseur des Films, den ich gestern gesehen habe?\nAssistent: Der Film, den Sie gestern gesehen haben, 'Inception', wurde von Christopher Nolan inszeniert.\nBenutzer: Welche anderen Filme hat er inszeniert?\n\nRewrite: Welche anderen Filme hat Christopher Nolan, der Regisseur von 'Inception', inszeniert?\n\n5. Beispiel:\nBenutzer: Wie heißt die Programmiersprache, die ich lerne?\nAssistent: Sie lernen Python.\nBenutzer: Wer hat es erstellt?\nAssistent: Python wurde von Guido van Rossum erstellt.\nBenutzer: Wann wurde es erstellt?\n\nRewrite: Wann wurde Python, die Programmiersprache, die von Guido van Rossum erstellt wurde, erstellt?<|im_end|>",
                    'string_dialogue': "<|im_start|user\nBitte formuliere die letzte Frage des Benutzers im Kontext um.<|im_end|>\n<|im_start|>assistant\n",
                    'user_prompt': 'Benutzer',
                    'assistant_prompt': 'Assistent'
                }
            },
            'en': {
                'GPT-3.5': {
                    'system_prompt': "As an assistant, you're tasked to decontextualize and reformulate the current question considering a multi-turn information-seeking dialogue. Please provide only the rewritten question after 'Rewrite:'. It will be used for passage retrieval. Here are some examples: \n\n 1. Example:\n[INST] Who is my favorite singer? [/INST]\nJustin Bieber\n[INST] When was he born? [/INST]\n\nRewrite: When was Justin Bieber born?\n\n2. Example:\n[INST] From which city in Germany is KSC the football club of? [/INST]\nKarlsruhe\n[INST] How many people live there? [/INST]\n\nRewrite: How many people live in Karlsruhe?\n\n3. Example:\n[INST] What's the name of the book I'm reading? [/INST]\nThe book you're reading is '1984' by George Orwell.\n[INST] Who is the main character? [/INST]\n\nRewrite: Who is the main character in '1984' by George Orwell?\n\n4. Example:\n[INST] What's the name of the director of the movie I watched yesterday? [/INST]\nThe movie you watched yesterday, 'Inception', was directed by Christopher Nolan.\n[INST] What other movies has he directed? [/INST]\n\nRewrite: What other movies has Christopher Nolan, the director of 'Inception', directed?\n\n5. Example:\n[INST] What's the name of the programming language I'm learning? [/INST]\nYou're learning Python.\n[INST] Who created it? [/INST]\nPython was created by Guido van Rossum.\n[INST] When was it created? [/INST]\n\nRewrite: When was Python, the programming language created by Guido van Rossum, created?",
                    'string_dialogue': "[INST] Please reformulate the user's last question in context. [/INST]",
                    'user_prompt': 'User',
                    'assistant_prompt': 'Assistant'
                },
                'Llama2-7B-chat': {
                    'system_prompt': "As an assistant, you're tasked to decontextualize and reformulate the current question considering a multi-turn information-seeking dialogue. Please provide only the rewritten question after 'Rewrite:'. It will be used for passage retrieval. Here are some examples: \n\n 1. Example:\n[INST] Who is my favorite singer? [/INST]\nJustin Bieber\n[INST] When was he born? [/INST]\n\nRewrite: When was Justin Bieber born?\n\n2. Example:\n[INST] From which city in Germany is KSC the football club of? [/INST]\nKarlsruhe\n[INST] How many people live there? [/INST]\n\nRewrite: How many people live in Karlsruhe?\n\n3. Example:\n[INST] What's the name of the book I'm reading? [/INST]\nThe book you're reading is '1984' by George Orwell.\n[INST] Who is the main character? [/INST]\n\nRewrite: Who is the main character in '1984' by George Orwell?\n\n4. Example:\n[INST] What's the name of the director of the movie I watched yesterday? [/INST]\nThe movie you watched yesterday, 'Inception', was directed by Christopher Nolan.\n[INST] What other movies has he directed? [/INST]\n\nRewrite: What other movies has Christopher Nolan, the director of 'Inception', directed?\n\n5. Example:\n[INST] What's the name of the programming language I'm learning? [/INST]\nYou're learning Python.\n[INST] Who created it? [/INST]\nPython was created by Guido van Rossum.\n[INST] When was it created? [/INST]\n\nRewrite: When was Python, the programming language created by Guido van Rossum, created?",
                    'string_dialogue': "[INST] Please reformulate the user's last question in context. [/INST]",
                    'user_prompt': 'User',
                    'assistant_prompt': 'Assistant'
                }
            }
        }

        system_prompt = prompts[st.session_state.selected_language][st.session_state.selected_model]['system_prompt']
        # string_dialogue += "\n\n"
        if st.session_state.selected_model == "Llama2-7B-chat":
            string_dialogue = ""

            for dict_message in st.session_state.messages[1:]:
                if dict_message["role"] == "user":
                    string_dialogue += f"[INST] " + dict_message["content"] + " [/INST]\n"
                else:
                    string_dialogue += dict_message["content"] + "\n"

            string_dialogue += prompts[st.session_state.selected_language][st.session_state.selected_model]['string_dialogue']
            string_dialogue += "\n\nRewrite:" 
            output = ""

            print(f"""
CQU:
system_prompt: {system_prompt}

string_dialogue: {string_dialogue}""")
        
            # The meta/llama-2-7b-chat model can stream output as it's running.
            output = replicate.run(
                "meta/llama-2-7b-chat",
                input={
                    "debug": False,
                    "top_k": 1,
                    "top_p": 0.1,
                    "prompt": string_dialogue,
                    "temperature": 0.3,
                    "system_prompt": system_prompt,
                    "max_new_tokens": 300,
                    "min_new_tokens": -1,
                    "repetition_penalty": 1,
                    "do_sample": True
                },
            )
            output = ''.join(output)

        elif st.session_state.selected_model == "GPT-3.5":

            # Use OpenAI's GPT-3.5 model

            if 'OPENAI_API_TOKEN' in st.secrets:
                openai_key = st.secrets['OPENAI_API_TOKEN']
            elif "OPENAI_API_TOKEN" in os.environ:
                openai_key = os.environ['OPENAI_API_TOKEN']

            client = OpenAI(api_key=openai_key)

            try:
                messages = [
                    {"role": "system", "content": prompts[st.session_state.selected_language][st.session_state.selected_model]['system_prompt'].replace("[INST]", "USER:").replace("[/INST]", "ASSISTANT:")},
                ]
                messages.extend(st.session_state.messages[1:])
                messages.append({"role": "user", "content": prompts[st.session_state.selected_language][st.session_state.selected_model]['string_dialogue'].replace("[INST] ", "").replace(" [/INST]", "")})

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                )

                output = response['choices'][0]['message']['content']

            except openai.RateLimitError as e:
                # Handle rate limit error gracefully
                print(f"Rate limit exceeded. Waiting for 20 seconds before retrying.")
                time.sleep(20)  # You may adjust the sleep duration
                # Retry the request or handle in a way that suits your application

            except Exception as e:
                # Handle other exceptions if needed
                print(f"An error occurred with OpenAI API: {e}")
                    # Handle the error in a way that suits your application
                
            output = response.choices[0].message.content
        
        elif st.session_state.selected_model == "leo-hessian-7B-chat":
            string_dialogue = ""

            for dict_message in st.session_state.messages[1:]:
                if dict_message["role"] == "user":
                    string_dialogue += f"<|im_start|>user\n" + dict_message["content"] + "<|im_end|>\n"
                else:
                    string_dialogue += "<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>\n"

            string_dialogue += prompts[st.session_state.selected_language][st.session_state.selected_model]['string_dialogue']
            string_dialogue += "Rewrite:" 
            output = ""
            print(f"""
CQU:
system_prompt: {system_prompt}

string_dialogue: {string_dialogue}""")


            headers = {
                "Accept" : "application/json",
                "Content-Type": "application/json" 
            }

            def query(payload):
                response = requests.post(st.session_state['leo_api_url'], headers=headers, json=payload)
                return response.json()
            
            inputs = system_prompt + "\n" + string_dialogue

            output = query({
                "inputs": inputs,
                "parameters": {
                    "top_k": 60,
                    "top_p": 0.95,
                    "temperature": 0.9,
                    "do_sample": True,
                    "max_new_tokens": 300
                }
            })

            output = output[0]["generated_text"]

        print(f"CQU output: {output}") 
        st.session_state.cqu = output.split(":")[-1]
        return output.split(":")[-1]

    def generate_chat_response():
        prompts = {
            'de' : {
                'GPT-3.5': {
                    'system_prompt': "Sie sind ein hilfreicher Assistent, der einem Studenten bei Fragen zu seinen Prüfungsordnungen an der Universität Heidelberg hilft. Sie generieren eine Antwort auf der Grundlage mehrerer Kontexte, die nicht unbedingt die Antwort enthalten. Bitte generieren Sie eine Antwort oder sagen Sie 'Ich kann diese Frage nicht beantworten', wenn Sie auf der Grundlage des bereitgestellten Kontexts keine Antwort geben können.",
                    'dialoge': 'Dialog',
                    'context': 'Kontext',
                    'gap': 'Kontext zur Beantwortung der Frage:'
                },
                'leo-hessian-7B-chat': {
                    'system_prompt': "<|im_start|>system\nSie sind ein hilfreicher Assistent, der einem Studenten bei Fragen zu seinen Prüfungsordnungen an der Universität Heidelberg hilft. Sie generieren eine Antwort auf der Grundlage mehrerer Kontexte, die nicht unbedingt die Antwort enthalten. Bitte generieren Sie eine Antwort oder sagen Sie 'Ich kann diese Frage nicht beantworten', wenn Sie auf der Grundlage des bereitgestellten Kontexts keine Antwort geben können.<|im_end|>",
                    'dialoge': 'Dialog',
                    'context': 'Kontext',
                    'gap': 'Kontext zur Beantwortung der Frage:'
                },
            },
            'en':{
                'GPT-3.5': {
                    'system_prompt': "You are an assistant helping a student with examination regulations of the Heidelberg University. Generate a concise answer based on the provided contexts. If the answer isn't in the contexts, state 'I can't answer this question'. If there's ambiguity, ask clarifying questions.",
                    'dialoge': 'Dialogue',
                    'context': 'Context',
                    'gap': 'Context to answer the question:'
                },
                'Llama2-7B-chat': {
                    'system_prompt': "You are an assistant helping a student with examination regulations of the Heidelberg University. Generate a concise answer based on the provided contexts. If the answer isn't in the contexts, state 'I can't answer this question'. If there's ambiguity, ask clarifying questions.",
                    'dialoge': 'Dialogue',
                    'context': 'Context',
                    'gap': 'Context to answer the question:'
                }
            }
        }

        system_prompt = prompts[st.session_state.selected_language][st.session_state.selected_model]['system_prompt']

        if st.session_state.selected_model == "Llama2-7B-chat":
            string_dialogue = f"{prompts[st.session_state.selected_language][st.session_state.selected_model]['dialoge']}:\n\n"

            for dict_message in st.session_state.messages[1:]:
                if dict_message["role"] == "user":
                    string_dialogue += f"[INST] " + dict_message["content"] + " [/INST]\n"
                else:
                    string_dialogue += dict_message["content"] + "\n"
            
            string_dialogue += "\n\n"
            string_dialogue += f"{prompts[st.session_state.selected_language][st.session_state.selected_model]['gap']}\n\n"

            for index, passage in enumerate(st.session_state.evidence[0][:st.session_state.selected_context_length]):
                string_dialogue += f"{index+1}. {prompts[st.session_state.selected_language][st.session_state.selected_model]['context']}:\n"
                string_dialogue += passage + "\n\n"
            
            output = ""
            print(f"""
CHAT:
system_prompt: {system_prompt}

string_dialogue: {string_dialogue}""") 

            # The meta/llama-2-7b-chat model can stream output as it's running.
            output = replicate.run(
                "meta/llama-2-7b-chat",
                input={
                    "debug": False,
                    "top_k": 5,
                    "top_p": 0.4,
                    "prompt": string_dialogue,
                    "temperature": 0.5,  # Lower temperature
                    "system_prompt": system_prompt,
                    "max_new_tokens": 300,  # Lower max tokens
                    "min_new_tokens": -1,
                    "repetition_penalty": 1.2  # Higher repetition penalty
                },
            )
            output = ''.join(output)
        
        elif st.session_state.selected_model == "GPT-3.5":

            if 'OPENAI_API_TOKEN' in st.secrets:
                openai_key = st.secrets['OPENAI_API_TOKEN']
            elif "OPENAI_API_TOKEN" in os.environ:
                openai_key = os.environ['OPENAI_API_TOKEN']

            client = OpenAI(api_key=openai_key)

            try:

                string_dialogue = f"{prompts[st.session_state.selected_language][st.session_state.selected_model]['gap']}\n\n"

                for index, passage in enumerate(st.session_state.evidence[0][:st.session_state.selected_context_length]):
                    string_dialogue += f"{index+1}. {prompts[st.session_state.selected_language][st.session_state.selected_model]['context']}:\n"
                    string_dialogue += passage + "\n\n"
            

                messages = [
                    {"role": "system", "content": prompts[st.session_state.selected_language][st.session_state.selected_model]['system_prompt'].replace("[INST]", "USER:").replace("[/INST]", "ASSISTANT:")},
                ]
                messages.extend(st.session_state.messages[1:])
                messages.append({"role": "user", "content": string_dialogue})

                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=messages,
                )

                output = response['choices'][0]['message']['content']

            except openai.RateLimitError as e:
                # Handle rate limit error gracefully
                print(f"Rate limit exceeded. Waiting for 20 seconds before retrying.")
                time.sleep(20)  # You may adjust the sleep duration
                # Retry the request or handle in a way that suits your application

            except Exception as e:
                # Handle other exceptions if needed
                print(f"An error occurred with OpenAI API: {e}")
                    # Handle the error in a way that suits your application
                
            output = response.choices[0].message.content
        
        elif st.session_state.selected_model == "leo-hessian-7B-chat":
            string_dialogue = ""

            for dict_message in st.session_state.messages[1:]:
                if dict_message["role"] == "user":
                    string_dialogue += f"<|im_start|>user\n" + dict_message["content"] + "<|im_end|>\n"
                else:
                    string_dialogue += "<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>\n"

            string_dialogue += f"<|im_start|>user\n{prompts[st.session_state.selected_language][st.session_state.selected_model]['gap']}\n\n"

            for index, passage in enumerate(st.session_state.evidence[0][:st.session_state.selected_context_length]):
                string_dialogue += f"{index+1}. {prompts[st.session_state.selected_language][st.session_state.selected_model]['context']}:\n"
                string_dialogue += passage + "\n\n"
            
            string_dialogue += "<|im_end|>\n<|im_start|>assistant\n"

            output = ""
            print(f"""
CHAT:
system_prompt: {system_prompt}
string_dialogue: {string_dialogue}""")

            headers = {
                "Accept" : "application/json",
                "Content-Type": "application/json" 
            }

            def query(payload):
                response = requests.post(st.session_state['leo_api_url'], headers=headers, json=payload)
                return response.json()
            
            inputs = system_prompt + "\n" + string_dialogue

            output = query({
                # "inputs": string_dialogue,
                "inputs": inputs,
                "parameters": {
                    "top_k": 10,
                    "top_p": 0.6,
                    "temperature": 0.4,
                    "max_new_tokens": 300,
                    # "repetition_penalty": 1.2,
                    "do_sample": True,
                    # "system_prompt": system_prompt,
                }
            })

            print("Output: ", output)

            output = output[0]["generated_text"]


        print(f"CHAT output: {output}")
        return output

    # User-provided prompt
    if prompt := st.chat_input(disabled=not st.session_state.replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("CQU generation..."):
                st.session_state.cqu = generate_cqu_response()
            with st.spinner("Evidence retrieval..."):
                evidence = retrieve_evidence()
            with st.spinner("Chat generation..."):
                response = generate_chat_response() 

                # Add row to history
                # Join evidence to a string with \n\n between them
                row = {'turn': len(st.session_state.history), 'question': st.session_state.messages[-1]["content"], 'answer': response, 'evidence': str("\n\n".join(evidence[0][:st.session_state.selected_context_length])), 'cqu': st.session_state.cqu, 'context_relevance': False, 'context_position': 0, 'faithfullness': False, 'answer_relevance': 0.0, 'reason_for_incorrectness':None}
                df_dictionary = pd.DataFrame([row])
                st.session_state.history = pd.concat([st.session_state.history, df_dictionary], ignore_index=True)

                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)
        st.session_state.update = not st.session_state.update 

else:
    # Display Data Editor for Evaluation
    st.title("📊 Evaluation")
    st.write("This is the data editor for the evaluation of the chatbot.")

    st.write("""
Explanation of the columns:
- **Turn**: The turn number of the conversation
- **Question (User)**: The question asked by the user
- **Answer (Assistant)**: The answer generated by the assistant
- **Retrieved Evidence**: The retrieved evidence
- **Contextualized Question**: The contextualized question generated by the assistant
- **Context Relevance**: Is the retrieved evidence relevant to the question? Yes(X) No()
- **Context Position**: At which position is the answer in the retrieved evidence?
- **Faithfullness**: Is the answer based on the facts from the Evidence? Yes(X) No()
- **Answer Relevance**: How relevant is the answer to the question? (1 - 10)
- **Reason for Incorrectness**: Why is the answer incorrect?
""")


    # history = st.session_state.history
    # Display editable dataframe of history 
    edited_data = st.data_editor(
        st.session_state.history,
        column_config={
            "turn": "Turn",
            "question": "Question (User)",
            "answer": "Answer (Assistant)",
            "evidence": "Retrieved Evidence",
            "cqu": st.column_config.TextColumn(
                "Contextualized Question",
                help="The contextualized question generated by the assistant",
            ),
            "context_relevance": st.column_config.CheckboxColumn(
                "Context Relevance",
                help="Is the retrieved evidence relevant to the question? Yes(X) No()",
                required=True,
            ),
            "context_position": st.column_config.NumberColumn(
                "Context Position",
                help="At which position is the answer in the retrieved evidence?",
                min_value=0,
                max_value=st.session_state.selected_context_length,
                step=1,
                required=True,
            ),
            "faithfullness": st.column_config.CheckboxColumn(
                "Faithfullness",
                help="Is the answer based on the facts from the Evidence? Yes(X) No()",
                required=True,
            ),
            "answer_relevance": st.column_config.NumberColumn(
                "Answer Relevance",
                help="How relevant is the answer to the question? (1 - 10)",
                min_value=0,
                max_value=10,
                step=1,
                required=True,
            ),
            "question_type": st.column_config.SelectboxColumn(
                "Question Type",
                help="What type of question is it?",
                width="medium",
                options=[
                    "Factoid",
                    "List",
                    "Yes/No",
                    "Abstract",
                    "Complex",
                    "Other"
                ],
            ),
            "reason_for_incorrectness": st.column_config.SelectboxColumn(
                "Reason for Incorrectness",
                help="Why is the answer incorrect?",
                width="medium",
                options=[
                    "Hallucination",
                    "Nonsense Answer",
                    "No Negative Rejection",
                    "Correference Error",
                    "Wrong Answer",
                    "Correct Evidence Not Found",
                    "Other"
                ],
            )
        },
        disabled=["turn", "question", "answer", "evidence", "cqu"],
        hide_index=True 
    )

    evaluator = st.selectbox('Evaluator', ('A', 'B', "Other"))

    # Create MongoDB Object
    upload_data = {
        "evaluator": evaluator, 
        "model": st.session_state.selected_model,
        "language": st.session_state.selected_language,
        "index": st.session_state.selected_index,
        "context_length": st.session_state.selected_context_length,
        "history": edited_data,
        "timestamp": pd.Timestamp.now()
    }
    
    if st.button("Submit"):
        # Divide the values in the upload_data history answer_relevance by 10
        upload_data["history"]["answer_relevance"] = upload_data["history"]["answer_relevance"] / 10

        # Convert the upload_data history to a dictionary
        upload_data["history"] = upload_data["history"].to_dict(orient="records")

        print("Submitted\n")

        # Upload data to MongoDB
        from pymongo import MongoClient

        # Get Mongo Name
        if 'REPLICATE_API_TOKEN' in st.secrets:
            st.session_state['replicate_api'] = st.secrets['REPLICATE_API_TOKEN']
        elif "REPLICATE_API_TOKEN" in os.environ:
            st.session_state['replicate_api'] = os.environ['REPLICATE_API_TOKEN']
        
        if "MONGO_DB_NAME" in st.secrets:
            mongo_name = st.secrets['MONGO_DB_NAME']
        elif "MONGO_DB_NAME" in os.environ:
            mongo_name = os.environ['MONGO_DB_NAME']

        if "MONDO_DB_PASSWORD" in st.secrets:
            mongo_password = st.secrets['MONDO_DB_PASSWORD']
        elif "MONDO_DB_PASSWORD" in os.environ:
            mongo_password = os.environ['MONDO_DB_PASSWORD']
        
        if "MONGO_DB_URL" in st.secrets:
            mongo_url = st.secrets['MONGO_DB_URL']
        elif "MONGO_DB_URL" in os.environ:
            mongo_url = os.environ['MONGO_DB_URL']
        
        if "MONGO_DB_DBNAME" in st.secrets:
            mongo_db_name = st.secrets['MONGO_DB_DBNAME']
        elif "MONGO_DB_DBNAME" in os.environ:
            mongo_db_name = os.environ['MONGO_DB_DBNAME']
        
        if "MONGO_DB_COLLECTION" in st.secrets:
            mongo_collection = st.secrets['MONGO_DB_COLLECTION']
        elif "MONGO_DB_COLLECTION" in os.environ:
            mongo_collection = os.environ['MONGO_DB_COLLECTION']

        client = MongoClient(f"mongodb+srv://{mongo_name}:{mongo_password}@{mongo_url}")

        db = client[mongo_db_name]
        collection = db[mongo_collection]

        with st.spinner("Uploading to MongoDB..."):
            collection.insert_one(upload_data)