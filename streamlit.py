import streamlit as st
import pandas as pd
import openai

openai.api_key = ""

s = pd.read_csv('startups.csv')

tab1, tab2 = st.tabs(['Cбор информации', 'Генерация лого'])
with tab1:
    st.title('Work Brothers Production')
    st.title('Название стартапа')
    company_name = st.text_input('              ')
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
    st.title('Краткое описание стартапа')
    short_description = st.text_area('')
    st.title('Проблема, которую решает стартап')
    problem_description = st.text_area('', height=200)
    st.title('Описание и ценность')
    value_and_description = st.text_area(' ')
    st.title('Решение')
    solution = st.text_area('  ')
    st.title('Рынок')
    market_option = st.selectbox('   ', ('РФ', 'Международный рынок'))
    tam = '''TAM: 

            ...'''
    st.markdown(tam)
    sam = st.text_area('SAM: ')
    som = st.text_area('SOM: ')
    st.title('Конкуренты')
    competitors = '''
    Список конкурентов:
    
        1. ...
         
        2. ... 
         
        3. ... 
         
         '''
    advantages = '''Конкурентные преимущества: 
    
    ...'''
    st.markdown(competitors)
    st.markdown(advantages)
    st.title('Бизнес модель  и ценообразование')
    business_model_1 = st.text_area('                          ')
    business_model_2 = st.text_area('                           ')
    business_model_3 = st.text_area('                         ')
    st.title('Трекшен и финансы')
    year_earnings = st.number_input('Годовая выручка (млн. руб.): ')
    clients_amount = st.number_input('Количество клиентов: ')
    metrics = '''
    1. APRU = ...
    
    2. Churn Rate = ...
    
    3. LT = ...
    
    4. LTV = ...
    '''
    st.markdown(metrics)
    st.title('Команда и борд')
    team = st.text_area('    ')
    st.title('Инвестиционный раунд')
    invest_round = st.text_area('     ')
    st.title('Дорожная карта')
    roadmap = st.text_area('      ')
    st.title('Контакты')
    contacts = st.text_area('       ')
    st.button("Создать неповторимый Pitch-Deck", type='primary')

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
