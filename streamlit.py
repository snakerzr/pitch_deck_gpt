import streamlit as st
import pandas as pd
import openai
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
                              'contacts_gpt':'','':'','':'',}

for key, value in default_session_state_dict.items():
    if key not in st.session_state:
        st.session_state[key] = value



s = pd.read_csv('data/startups.csv')

pa = PitchAssistant()

tab1, tab2 = st.tabs(['Cбор информации', 'Генерация лого'])
with tab1:
    st.title('Work Brothers Production')
    st.title('Название стартапа')
    company_name = st.text_input('              ', key='company_name')
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
            st.write(text[0].upper() + text[1:])

    # Краткое описание стартапа
    st.title('Краткое описание стартапа')
    short_description = st.text_area('',key='short_description')

    if st.button("Готово", type='primary', key=1): 
        short_description_gpt = pa.short_description(short_description)
        st.session_state['short_description_gpt'] = short_description_gpt
    st.markdown(st.session_state['short_description_gpt'])

    # Проблема, которую решает стартап
    st.title('Проблема, которую решает стартап')
    problem_description = st.text_area('', height=200, key='problem_description')

    if st.button("Готово", type='primary', key=2):
        problem_description_gpt = pa.problem(problem_description)
        st.session_state['problem_description_gpt'] = problem_description_gpt
    st.markdown(st.session_state['problem_description_gpt'])

    # Описание и ценность
    st.title('Описание и ценность')
    value_and_description = st.text_area('', key='value_and_description')

    if st.button("Готово", type='primary', key=3): 
        value_and_description_gpt = pa.value(value_and_description)
        st.session_state['value_and_description_gpt'] = value_and_description_gpt
    st.markdown(st.session_state['value_and_description_gpt'])

    # Решение
    st.title('Решение')
    solution = st.text_area('', key='solution')

    if st.button("Готово", type='primary', key=4):
        solution_gpt = pa.solution(solution)
        st.session_state['solution_gpt'] = solution_gpt
    st.markdown(st.session_state['solution_gpt'])

    # Рынок
    st.title('Рынок')
    market_option = st.selectbox('   ', ('РФ', 'Международный рынок'))
    if st.button("Готово", type='primary', key=5):
        # tam = pa.market_tom(value_and_description)
        tam = pa.market_tam(st.session_state['value_and_description_gpt'])
        st.session_state['tam'] = tam
        tam_summary = pa.market_tam_summary(tam)
        st.session_state['tam_summary'] = tam_summary

        # sam_instruction = pa.market_sam_instruction(value_and_description)
        sam_instruction = pa.market_sam_instruction(st.session_state['value_and_description_gpt'])
        st.session_state['sam_instruction'] = sam_instruction


        # som_instruction = pa.market_som_instruction(value_and_description)
        som_instruction = pa.market_som_instruction(st.session_state['value_and_description_gpt'])
        st.session_state['som_instruction'] = som_instruction

    
    st.markdown(st.session_state['tam'])
    st.markdown(st.session_state['tam_summary'])
    st.markdown(st.session_state['sam_instruction'])
    sam = st.text_area('SAM: ', key='sam')
    st.markdown(st.session_state['som_instruction'])
    som = st.text_area('SOM: ', key='som')


    # Конкуренты
    st.title('Конкуренты')
    competitors = '''
    Список конкурентов:
    
        1. ...
         
        2. ... 
         
        3. ... 
         
         '''
    advantages = '''Конкурентные преимущества: 
    
    ...'''

    if st.button("Готово", type='primary', key=6): 
        competitors_list_gpt = pa.competitors_list(st.session_state['short_description_gpt'])
        st.session_state['competitors_list_gpt'] = competitors_list_gpt
        competitive_advantages_gpt = pa.competitive_advantages(st.session_state['value_and_description_gpt'],
                                                               st.session_state['competitors_list_gpt'])
        st.session_state['competitive_advantages_gpt'] = competitive_advantages_gpt

 
    st.markdown(st.session_state['competitors_list_gpt'])
    st.markdown(st.session_state['competitive_advantages_gpt'])


    # Бизнес модель  и ценообразование
    st.title('Бизнес модель  и ценообразование')

    if st.button("Начать", type='primary', key=7):
        # Вопрос: Напишите способ генерации дохода и основные затраты вашего бизнеса?
        # Сначала отвечает чатгпт, дает пример, потом стартапер пишет свой
        profit_and_costs_example = pa.bis_model_profit_generation_and_costs(st.session_state['value_and_description_gpt'])
        st.session_state['profit_and_costs_example'] = profit_and_costs_example
        st.markdown(st.session_state['profit_and_costs_example'])
        business_model_1 = st.text_area('', key='business_model_1')

        # Вопрос: Какие каналы вы используете для привлечения клиентов?
        # 
        acquisition_example = pa.bis_model_acquisition_channels(st.session_state['value_and_description_gpt'])
        st.session_state['acquisition_example'] = acquisition_example
        st.markdown(st.session_state['acquisition_example'])
        business_model_2 = st.text_area('', key='business_model_2')

        # Вопрос: Как вы видите идите масштабирование своей бизнес-модели?
        # 
        bis_model_scaling_example = pa.bis_model_scaling(st.session_state['value_and_description_gpt'])
        st.session_state['bis_model_scaling_example'] = bis_model_scaling_example
        st.markdown(st.session_state['bis_model_scaling_example'])
        business_model_3 = st.text_area('', key='business_model_3')


    # Трекшн и финансы
    st.title('Трекшен и финансы')
    year_earnings = st.number_input('Годовая выручка (млн. руб.): ')
    clients_amount = st.number_input('Количество клиентов: ')
    if st.button("Готово", type='primary', key=8):
        metrics = pa.fin_metrics(st.session_state['value_and_description_gpt'],
                                    year_earnings,
                                    clients_amount)
        st.session_state['metrics'] = metrics
            
    st.markdown(st.session_state['metrics'])


    if st.button("Готово", type='primary', key=9):
        traction_and_partners_example = pa.traction_and_partners(st.session_state['value_and_description_gpt'])
        st.session_state['traction_and_partners_example'] = traction_and_partners_example
    
    st.markdown(st.session_state['traction_and_partners_example'])
    traction_and_partners = st.text_area('', key='traction_and_partners')

    # Команда и борд
    st.title('Команда и борд')
    team = st.text_area('', key='team')
    if st.button("Готово", type='primary', key=10):
        team_gpt_example = pa.team_board(team)
        st.session_state['team_gpt_example'] = team_gpt_example
        st.markdown(st.session_state['team_gpt_example'])
        
    

    # Инвестиционный раунд
    st.title('Инвестиционный раунд')
    investment_round = st.text_area('', key='investment_round')
    if st.button("Готово", type='primary', key=11):
        investment_round_example = pa.investment_round(investment_round)
        st.session_state['investment_round_example'] = investment_round_example
        st.markdown(st.session_state['investment_round_example'])


    # Дорожная карта
    st.title('Дорожная карта')
    roadmap = st.text_area('', key='roadmap')
    if st.button("Готово", type='primary', key=12):
        roadmap_gpt = pa.roadmap(roadmap)
        st.session_state['roadmap_gpt'] = roadmap_gpt
        st.markdown(st.session_state['roadmap_gpt'])

    # Контакты
    st.title('Контакты')
    contacts = st.text_area('', key='contacts')
    if st.button("Готово", type='primary', key=13):
        contacts_gpt = pa.contaсts(contacts)
        st.session_state['contacts_gpt'] = contacts_gpt
        st.markdown(st.session_state['contacts_gpt'])

    all_is_ready = st.button("Создать неповторимый Pitch-Deck", type='primary')
    if st.button("Готово", type='primary', key=14):
        # pa.create_presentation_text()
        pass

with tab2:
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
