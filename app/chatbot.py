from operator import itemgetter
import os

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_models import ChatYandexGPT
from langchain_community.embeddings import GigaChatEmbeddings, YandexGPTEmbeddings
from langchain_community.chat_models import GigaChat
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.globals import set_debug

# set_debug(True)

from promts import main_promt


import streamlit as st
import os

# from generate_token import generate_token

# это основная функция, которая запускает приложение streamlit
def main():
    # Загрузка логотипа компании
    logo_image = 'app/images/logo-2.jpg'  # Путь к изображению логотипа

    # # # Отображение логотипа в основной части приложения
    from PIL import Image
    # Загрузка логотипа
    logo = Image.open(logo_image)
    # Изменение размера логотипа
    resized_logo = logo.resize((200, 150))
    st.set_page_config(page_title="PetPalsGPT", page_icon="📖")   
    # Отображаем лого измененного небольшого размера
    st.image(resized_logo)
    st.title('📖 PetPalsGPT')
    """
    Чатбот на базе GPT, который запоминает контекст беседы. Чтобы "сбросить" контекст обновите страницу браузера.\n
    Вы можете выбрать какую модель использовать.
    """
    # st.warning('Это Playground для общения с YandexGPT')

    # вводить все credentials в графическом интерфейсе слева
    # Sidebar contents
    with st.sidebar:
        st.title('\U0001F917\U0001F4AC PetPalsGPT чатбот')
        st.markdown('''
        ## О программе
        Данный чатбот использует следующие компоненты:
        - [Streamlit](https://streamlit.io/)
        - [LangChain](https://python.langchain.com/)
        ''')

    # Настраиваем алгоритмы работы памяти
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    if len(msgs.messages) == 0:
        msgs.add_ai_message("Привет! Я ваш виртуальный ассистент по вопросам о домашних животных. Чем могу помочь сегодня?")

    view_messages = st.expander("Просмотр истории сообщений")


    # Загрузка переменных из файла .env
    # load_dotenv()
    yc_folder_id = st.secrets.get("YC_FOLDER_ID")

    if not yc_folder_id:
        yc_folder_id = st.sidebar.text_input("YC folder ID", type="password")
        if not yc_folder_id:
            st.info("Укажите [YC folder ID](https://cloud.yandex.ru/ru/docs/yandexgpt/quickstart#yandex-account_1) для запуска чатбота")
            st.stop()

    
        
    yc_iam_token = st.sidebar.text_input("YC IAM token", type="password")
    if not yc_iam_token:
        st.info("Укажите [YC IAM token](https://cloud.yandex.ru/ru/docs/yandexgpt/quickstart#yandex-account_1) для запуска чатбота")
        st.stop()
        
    
    # yagpt_folder_id = st.secrets["YC_FOLDER_ID"]
    # yc_service_account_id = st.secrets["YC_SERVICE_ACCOUNT_ID"]
    # yc_key_id = st.secrets["YC_KEY_ID"]
    # yc_private_key = st.secrets["YC_PRIVATE_KEY"]
    # yc_private_key = load_private_key(".streamlit/private_key.pem", "your_password")
    # yc_iam_token = generate_token('.streamlit/sa-gen-petpals-gpt-00001.json')
    gigachat_credintials = st.secrets["GIGACHAT_CREDENTIALS"]
     

    # with st.sidebar:
    #     st.markdown('''
    #         ## Дополнительные настройки
    #         Можно выбрать модель, степень креативности и системный промпт
    #         ''')

    model_dict = {
      "YandexGPT Lite": "gpt://b1gr0nm9o4sp7b51etoh/yandexgpt-lite/latest",
      "YandexGPT Lite RC": "gpt://b1gr0nm9o4sp7b51etoh/yandexgpt-lite/rc",
      "YandexGPT Pro": "gpt://b1gr0nm9o4sp7b51etoh/yandexgpt/latest",
      "GigaChat LIte": "GigaChat",
      "GigaChat Lite+": "GigaChat-Plus",
      "GigaChat Pro": "GigaChat-Pro",
    }
    index_model = 0
    # selected_model = st.sidebar.radio("Выберите модель для работы:", model_dict.keys(), index=index_model, key="index" ) 
    selected_model = "YandexGPT Pro"
    
    # yagpt_prompt = st.sidebar.text_input("Промпт-инструкция для YaGPT")
    # Добавляем виджет для выбора опции
    # prompt_option = st.sidebar.selectbox(
    #     'Выберите какой системный промпт использовать',
    #     ('По умолчанию', 
    #      #'Задать самостоятельно'
    #      )
    
    default_prompt = main_promt
    # # Если выбрана опция "Задать самостоятельно", показываем поле для ввода промпта
    # if prompt_option == 'Задать самостоятельно':
    #     custom_prompt = st.sidebar.text_input('Введите пользовательский промпт:')
    # else:
    #     custom_prompt = default_prompt
    #     # st.sidebar.write(custom_prompt)
    #     with st.sidebar:
    #         st.code(custom_prompt)
    # # Если выбрали "задать самостоятельно" и не задали, то берем дефолтный промпт
    # if len(custom_prompt)==0: custom_prompt = default_prompt
    custom_prompt = default_prompt

    # temperature = st.sidebar.slider("Степень креативности (температура)", 0.0, 1.0, 0.6)
    temperature = 0.3
    # yagpt_max_tokens = st.sidebar.slider("Размер контекстного окна (в [токенах](https://cloud.yandex.ru/ru/docs/yandexgpt/concepts/tokens))", 200, 8000, 5000)

    def history_reset_function():
        # Code to be executed when the reset button is clicked
        st.session_state.clear()

    st.sidebar.button("Обнулить историю общения", on_click=history_reset_function)
    
    # Настраиваем LangChain, передавая Message History
    # промпт с учетом контекста общения

    if selected_model.startswith("GigaChat"):
        model = GigaChat(
            credentials=gigachat_credintials,
            verify_ssl_certs=False, scope="GIGACHAT_API_CORP", name=selected_model, temperature = temperature, #max_tokens = yagpt_max_tokens
                         )
        embeddings = GigaChatEmbeddings(verify_ssl_certs=False, scope="GIGACHAT_API_CORP")
        index = FAISS.load_local("app/index/index_chunk_1000-chars_embeddings_giga_chat", embeddings, allow_dangerous_deserialization=True)
    else:
        model_uri = model_dict[selected_model]
        model = ChatYandexGPT(
            iam_token=yc_iam_token,
            model_uri=model_uri,
            temperature=temperature,  
            # max_tokens = yagpt_max_tokens
        )
        emb_model_uri = "emb://b1gr0nm9o4sp7b51etoh/text-search-doc/latest"
        embeddings = YandexGPTEmbeddings(
            iam_token=yc_iam_token, model_uri=emb_model_uri, folder_id=yc_folder_id
        )
        index = FAISS.load_local(
            "app/index/index_chunk_1000-chars_embeddings_ya-text-search-doc-v2",
            embeddings,
            allow_dangerous_deserialization=True,
        )

    
    combined_prompt = ChatPromptTemplate.from_messages([
    ("system", custom_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", """
    Пожалуйста, посмотри на текст ниже и ответь на вопрос, используя информацию из этого текста, релевантную вопросу, а также учитывая предыдущий контекст разговора. Не выдумывай информацию.
    
    Текст:
    -----
    {context}
    -----
    
    Вопрос: {question}
    """)
    ])
    
    def get_question(input):
        if not input:
            return None
        elif isinstance(input,str):
            return input
        elif isinstance(input,dict) and 'question' in input:
            return input['question']
        elif isinstance(input,BaseMessage):
            return input.content
        else:
            raise Exception("string or dict with 'question' key expected as RAG chain input.")

    from langchain.schema import AIMessage, HumanMessage
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    retriever = index.as_retriever(search_kwargs = {"k": 10})

    # Создаем цепочку
    chain = (
        {
            "context": RunnableLambda(get_question) | retriever | format_docs, 
            "question": RunnablePassthrough(),
            "history": lambda x: x["history"]
        }
        | combined_prompt
        | model
        | StrOutputParser()
    )

    # Создаем цепочку с историей
    chain_with_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: msgs,
        input_messages_key="question",
        history_messages_key="history",
    )

    # Отображаем текущие сообщения из StreamlitChatMessageHistory
    for msg in msgs.messages:
        st.chat_message(msg.type).write(msg.content)

    # Если пользователь вводит новое приглашение, генерируем и отображаем новый ответ
    if prompt := st.chat_input():
        st.chat_message("human").write(prompt)
        config = {"configurable": {"session_id": "any"}}
        response = chain_with_history.invoke({"question": prompt}, config)
    
        # Проверяем тип response и соответственно обрабатываем
        if isinstance(response, AIMessage):
            st.chat_message("ai").write(response.content)
        elif isinstance(response, str):
            st.chat_message("ai").write(response)
        else:
            st.chat_message("ai").write(str(response))

    # Отобразить сообщения в конце, чтобы вновь сгенерированные отображались сразу
    with view_messages:
        """
        История сообщений, инициализированная с помощью:
        ```python
        msgs = StreamlitChatMessageHistory(key="langchain_messages")
        ```

        Содержание `st.session_state.langchain_messages`:
        """
        view_messages.json(st.session_state.langchain_messages)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.write(f"Что-то пошло не так. Возможно, не хватает входных данных для работы. {str(e)}")