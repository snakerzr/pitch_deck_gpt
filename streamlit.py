import os
import streamlit as st
import pandas as pd
import openai
from bs4 import BeautifulSoup
from tqdm import tqdm
import requests
from src.PitchAssistant import PitchAssistant

default_session_state_dict = {"short_description_gpt": '', 
                              "problem_description_gpt": '',
                              'value_and_description_gpt':'',
                              'solution_gpt':'',
                              'tam':'',
                              'tam_summary':'',
                              'sam_instruction':'',
                              'som_instruction':'',
                              'competitors_list_gpt':'',
                              'competitive_advantages_gpt':'',
                              'profit_and_costs_example':'',
                              'acquisition_example':'',
                              'bis_model_scaling_example':'',
                              'metrics':'',
                              'traction_and_partners_example':'',
                              'team_gpt_example':'',
                              'investment_round_example':'',
                              'roadmap_gpt':'',
                              'contacts_gpt':'',
                              'page_n':0,
                              'company_name':'',
                              'short_description':'',
                              'problem_description':'',
                              'value_and_description':'',
                              'solution':'',
                              'sam':'',
                              'som':'',
                              'business_model_1':'',
                              'business_model_2':'',
                              'business_model_3':'',
                              'year_earnings':'',
                              'clients_amount':'',
                              'traction_and_partners':'',
                              'team':'',
                              'investment_round':'',
                              'roadmap':'',
                              'contacts':'',
                              'presentation_text':''}

for key, value in default_session_state_dict.items():
    if key not in st.session_state:
        st.session_state[key] = value


# sidebar_radio_captions = ['Титульный лист',
#                           'Проблема',
#                           'Описание и ценностное предложение стартапа',
#                           'Решение',
#                           'Рынок',
#                           'Конкуренты',
#                           'Бизнес модель и ценообразование',
#                           'Трекшн и финансы',
#                           'Команда и борд',
#                           'Инвестиционный раунд',
#                           'Дорожная карта',
#                           'Контакты']
# with st.sidebar:
#     st.radio('Page n',
#              list(range(12)),
#             #  captions=sidebar_radio_captions,
#              key='page_n')

def parse_financial_indicators(link: str):
    response = requests.get(link)
    try:
        soup = BeautifulSoup(response.content)
    except:
        print(f"Ошибка запроса: {response.status_code}")

    financial_indicators = soup.find_all("span", {"class": "cCard__BlockMaskSum"})
    profitability_indicators = soup.find_all("div", {"class": "cCard__BlockRating cCard__Padding10"})
    # Выручка
    revenue = financial_indicators[0].text
    # Прибыль
    profit = financial_indicators[1].text
    # Стоимость компании
    company_cost = financial_indicators[2].text
    # рентабильность продаж
    sales = profitability_indicators[0].text
    # рентаьельность капитала
    capital = profitability_indicators[1].text

    return revenue, profit, company_cost, sales, capital

s = pd.read_csv('data/startups.csv')

pa = PitchAssistant()

tab1, tab2 = st.tabs(['Cбор информации', 'Генерация лого'])
with tab1:
    # if st.session_state['page_n'] == 0:
    # st.title('Work Brothers Production')
    st.title('Название стартапа')
    company_name = st.text_input('Введите название стартапа:', key='company_name')
    if st.button("Проверить", type='primary'):
        row = s[s['company'] == company_name.lower()]
        if len(row) == 0:
            st.subheader('Запись не найдена')
        else:
            company = str(*row['company'].values)
            market = str(*row['market'].values)
            market_ru = str(*row['market_ru'].values)
            about = str(*row['about'].values)
            description = str(*row['description'].values)
            earnings = str(*row['earnings'].values)
            stage = str(*row['stage'].values)
            inter_track = str(*row['inter_track'].values)
            company_link = str(*row['company_link'].values)
            sbis_link = str(*row['sbis_link'].values)
            text = description
            if earnings != '':
                if text[-1] != '.':
                    text += '.'
                text += ' Выручка компании ' + earnings + '.'
            if stage != '':
                if text[-1] != '.':
                    text += '.'
                text += ' Она находится на стадии ' + stage + '.'
            if inter_track:
                if text[-1] != '.':
                    text += '.'
                text += ' Компания нацелена на международный рынок.'
            try:
                revenue, profit, company_cost, sales, capital = parse_financial_indicators(sbis_link)
                if text[-1] != '.':
                    text += '.'
                text += ' Стоимость компании ' + str(company_cost) + '.'
            except:
                pass
            st.write(text[0].upper() + text[1:])

    # Краткое описание стартапа
    st.title('Краткое описание стартапа')
    st.caption('Введите краткое или подробное описание стартапа. Оно преобразуется в короткое, отражающее суть.')
    short_description = st.text_area('Введите краткое описание стартапа',key='short_description')

    if st.button("Преобразовать", type='primary', key=1): 
        short_description_gpt = pa.short_description(short_description)
        st.session_state['short_description_gpt'] = short_description_gpt
    st.markdown(st.session_state['short_description_gpt'])

    # elif st.session_state['page_n'] == 1:
    # Проблема, которую решает стартап
    st.title('Проблема, которую решает стартап')
    st.caption('Введите описание проблемы, которую решает стартап в свободной форме. Предпочтительно подробно.')
    problem_description = st.text_area('', height=200, key='problem_description')

    if st.button("Создать текст для слайда", type='primary', key=2):
        problem_description_gpt = pa.problem(problem_description)
        st.session_state['problem_description_gpt'] = problem_description_gpt
    st.markdown(st.session_state['problem_description_gpt'])

    # elif st.session_state['page_n'] == 2:

    # Описание и ценность
    st.title('Описание и ценностное предложение стартапа')
    st.caption('Введите в свободной форме подробно описание и ценностное предложение стартапа.')
    # st.markdown('''Пример''')
    value_and_description = st.text_area('', key='value_and_description')

    if st.button("Создать текст для слайда", type='primary', key=3): 
        value_and_description_gpt = pa.value(value_and_description)
        st.session_state['value_and_description_gpt'] = value_and_description_gpt
    st.markdown(st.session_state['value_and_description_gpt'])

    # elif st.session_state['page_n'] == 3:

    # Решение
    st.title('Решение')
    st.caption('Опишите подробно, в свободной форме функционал вашего решения. То, как он помогает закрыть потребности клиента.')
    solution = st.text_area('', key='solution')

    if st.button("Создать текст для слайда", type='primary', key=4):
        solution_gpt = pa.solution(solution)
        st.session_state['solution_gpt'] = solution_gpt
    st.markdown(st.session_state['solution_gpt'])

    # elif st.session_state['page_n'] == 4:

    # Рынок
    st.title('Рынок')
    st.markdown('''Здесь мы вместе рассчитаем TAM, SAM, SOM вашего стартапа.
                
                Инструкция:
                1. Выберите географию вашего стартапа
                2. Посмотрите на появившуюся информацию о TOM вашего рынка
                3. Пользуясь инструкциями заполните SAM вашего рынка
                4. Пользуясь инструкциями заполните SOM вашего рынка
                
                
                ''')

    st.caption('Пожалуйста, выберите по какому географическому признаку исследовать рынок: РФ или Международный.')
    market_option = st.selectbox('   ', ('РФ', 'Международный рынок'))
    if st.button("Провести исследование", type='primary', key=5):
        # tam = pa.market_tom(value_and_description)
        tam = pa.market_tam(st.session_state['short_description_gpt'],market_option)
        st.session_state['tam'] = tam
        tam_summary = pa.market_tam_summary(st.session_state['tam'])
        st.session_state['tam_summary'] = tam_summary

        # sam_instruction = pa.market_sam_instruction(value_and_description)
        sam_instruction = pa.market_sam_instruction(st.session_state['value_and_description_gpt'])
        st.session_state['sam_instruction'] = sam_instruction

        # som_instruction = pa.market_som_instruction(value_and_description)
        som_instruction = pa.market_som_instruction(st.session_state['value_and_description_gpt'])
        st.session_state['som_instruction'] = som_instruction

    st.caption('Здесь будет TOM вашего стартапа.')
    st.markdown(st.session_state['tam'])
    st.markdown(st.session_state['tam_summary'])

    st.subheader('SAM')
    st.markdown(st.session_state['sam_instruction'])
    sam = st.text_area('SAM: ', key='sam')

    st.subheader('SOM')
    st.markdown(st.session_state['som_instruction'])
    som = st.text_area('SOM: ', key='som')

    # elif st.session_state['page_n'] == 5:

    # Конкуренты
    st.title('Конкуренты')
    st.caption('Здесь автоматически находятся конкуренты и определяются ваши конкурентные преимущства')
    if st.button("Найти конкурентов и определить преимущества", type='primary', key=6): 
        competitors_list_gpt = pa.competitors_list(st.session_state['short_description_gpt'])
        st.session_state['competitors_list_gpt'] = competitors_list_gpt
        competitive_advantages_gpt = pa.competitive_advantages(st.session_state['value_and_description_gpt'],
                                                            st.session_state['competitors_list_gpt'])
        st.session_state['competitive_advantages_gpt'] = competitive_advantages_gpt


    st.markdown(st.session_state['competitors_list_gpt'])
    st.markdown(st.session_state['competitive_advantages_gpt'])


    # elif st.session_state['page_n'] == 6:

    # Бизнес модель  и ценообразование
    st.title('Бизнес модель  и ценообразование')
    st.caption('''Здесь мы вместе последовательно ответим на 3 вопроса, пользуясь предоставленным примером''')

    st.subheader('Опишите способ генерации дохода и основные затраты вашего бизнеса?')
    if st.button('Привести пример',key='bis_example_1'):
        profit_and_costs_example = pa.bis_model_profit_generation_and_costs(st.session_state['value_and_description_gpt'])
        st.session_state['profit_and_costs_example'] = profit_and_costs_example
    st.markdown(st.session_state['profit_and_costs_example'])
    business_model_1 = st.text_area('Способ генерации дохода и основные затраты', key='business_model_1')

    st.subheader('Какие каналы вы используете для привлечения клиентов?')
    if st.button('Привести пример',key='bis_example_2'):
        acquisition_example = pa.bis_model_acquisition_channels(st.session_state['value_and_description_gpt'])
        st.session_state['acquisition_example'] = acquisition_example
    st.markdown(st.session_state['acquisition_example'])
    business_model_2 = st.text_area('Каналы для привлечения клиентов', key='business_model_2')

    st.subheader('Как вы видите идите масштабирование своей бизнес-модели?')
    if st.button('Привести пример',key='bis_example_3'):
        profit_and_costs_example = pa.bis_model_profit_generation_and_costs(st.session_state['value_and_description_gpt'])
        st.session_state['profit_and_costs_example'] = profit_and_costs_example
    st.markdown(st.session_state['bis_model_scaling_example'])
    business_model_3 = st.text_area('Масштабирование бизнес-модели', key='business_model_3')       


    # elif st.session_state['page_n'] == 7:

    # Трекшн и финансы
    st.title('Трекшн и финансы')
    st.caption('Введите годовую вырочку и количество клиентов, чтобы расчитать ARPU, churn rate, LT, LTV')
    year_earnings = st.number_input('Годовая выручка (млн. руб.): ',key='year_earnings')
    clients_amount = st.number_input('Количество клиентов: ',key='clients_amount')
    if st.button("Рассчитать метрики", type='primary', key=8):
        metrics = pa.fin_metrics(st.session_state['value_and_description_gpt'],
                                    year_earnings,
                                    clients_amount)
        st.session_state['metrics'] = metrics
            
    st.markdown(st.session_state['metrics'])


    st.subheader('Опишите ваш трекшн и партнеров')
    if st.button("Привести пример", type='primary', key=9):
        traction_and_partners_example = pa.traction_and_partners(st.session_state['value_and_description_gpt'])
        st.session_state['traction_and_partners_example'] = traction_and_partners_example
    
    st.markdown(st.session_state['traction_and_partners_example'])
    traction_and_partners = st.text_area('Трекшн и партнеры', key='traction_and_partners')

    # elif st.session_state['page_n'] == 8:

    # Команда и борд
    st.title('Команда и борд')
    st.caption('Опишите команду вашего проекта, бекграунд, текущих инвесторов и эдвайзеров')
    team = st.text_area('', key='team')
    if st.button("Создать текст для слайда", type='primary', key=10):
        team_gpt_example = pa.team_board(team)
        st.session_state['team_gpt_example'] = team_gpt_example
    st.markdown(st.session_state['team_gpt_example'])

        
    
    # elif st.session_state['page_n'] == 9:

    # Инвестиционный раунд
    st.title('Инвестиционный раунд')
    st.caption('Пожалуйста, в свободной форме опишите на каком инвестиционном раунде вы находитесь и на какие цели привлекаете инвестиции')
    investment_round = st.text_area('', key='investment_round')
    if st.button("Создать текст для слайда", type='primary', key=11):
        investment_round_example = pa.investment_round(investment_round)
        st.session_state['investment_round_example'] = investment_round_example
    st.markdown(st.session_state['investment_round_example'])

    # elif st.session_state['page_n'] == 10:

    # Дорожная карта
    st.title('Дорожная карта')
    st.caption('Пожалуйста, опишите в свободной форме ваш роадмап и дедлайны.')
    roadmap = st.text_area('', key='roadmap')
    if st.button("Создать текст для слайда", type='primary', key=12):
        roadmap_gpt = pa.roadmap(roadmap)
        st.session_state['roadmap_gpt'] = roadmap_gpt
    st.markdown(st.session_state['roadmap_gpt'])

    # elif st.session_state['page_n'] == 11:

    # Контакты
    st.title('Контакты')
    st.caption('В свободной форме напишите контакты, которые вы бы хотели продемонстрировать в презентации.')
    contacts = st.text_area('', key='contacts')
    if st.button("Создать текст для слайда", type='primary', key=13):
        contacts_gpt = pa.contaсts(contacts)
        st.session_state['contacts_gpt'] = contacts_gpt
    st.markdown(st.session_state['contacts_gpt'])

    # all_is_ready = st.button("Создать неповторимый Pitch-Deck", type='primary')
    # if all_is_ready:
    #     st.session_state['presentation_text'] = pa.create_presentation_text(st.session_state)


    agent_button = st.button('Сделать презентацию')
    if agent_button:
        st.session_state['presentation_text'] = pa.create_presentation_text_agent(st.session_state)

    st.markdown(st.session_state['presentation_text'])

    # else:
    #     st.session_state['page_n'] = 0

    st.session_state
    # some_dict = st.session_state
    # for key in some_dict:
    #     st.write(key)

with tab2:
    openai.api_key = os.environ['OPENAI_API_KEY ']
    photo_prompt = st.text_input('Описание логотипа стартапа: ')
    if st.button("Создать логотип", type='primary'):
        response = openai.Image.create(
            prompt=photo_prompt,
            n=1,
            size="1024x1024",
        )
        st.write(response["data"][0]["url"])
        #st.write('[link](response["data"][0]["url"])')
        #st.write("check out this [link](https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py)")
