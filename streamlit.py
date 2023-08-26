import streamlit as st
import pandas as pd
import openai
import src.PitchAssistant as PitchAssistant

openai.api_key = ""

s = pd.read_csv('data/startups.csv')

pa = PitchAssistant()

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

    # IF FALSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if False: 
        short_description_gpt = pa.short_description(short_description)
        st.text(short_description_gpt)

    st.title('Проблема, которую решает стартап')
    problem_description = st.text_area('', height=200)

    # IF FALSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if False: 
        problem_description_gpt = pa.problem(problem_description)
        st.text(problem_description_gpt)


    st.title('Описание и ценность')
    value_and_description = st.text_area(' ')

    # IF FALSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if False: 
        value_and_description_gpt = pa.value(value_and_description)
        st.text(value_and_description_gpt)


    st.title('Решение')
    solution = st.text_area('  ')

    # IF FALSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if False: 
        solution_gpt = pa.solution(solution)
        st.text(solution_gpt)

    
    st.title('Рынок')
    market_option = st.selectbox('   ', ('Выберите географический тип рынка','РФ', 'Международный рынок'))
    if market_option != 'Выберите географический тип рынка':
            # # IF FALSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # if False: 
        # tam = pa.market_tom(value_and_description)
        tom = pa.market_tom(value_and_description_gpt)
        tom_summary = pa.market_tom_summary(tom)
        st.markdown(tam)

        # sam = pa.market_sam_instruction(value_and_description)
        sam_instruction = pa.market_sam_instruction(value_and_description_gpt)
        st.markdown(sam_instruction)
        sam = st.text_area('SAM: ')

        # som_instruction = pa.market_som_instruction(value_and_description)
        som_instruction = pa.market_som_instruction(value_and_description_gpt)
        st.markdown(som_instruction)
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

    # IF FALSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if False: 
        competitors_list_gpt = pa.competitors.list(short_description_gpt)
        competitive_advantages = pa.competitive_advantages(value_and_description_gpt,competitors_list_gpt)

 
    st.markdown(competitors)
    st.markdown(advantages)



    st.title('Бизнес модель  и ценообразование')
    # IF FALSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if False: 
        # Вопрос: Напишите способ генерации дохода и основные затраты вашего бизнеса?
        # Сначала отвечает чатгпт, дает пример, потом стартапер пишет свой
        profit_and_costs_example = pa.bis_model_profit_generation_and_costs(value_and_description_gpt)
        st.markdown(profit_and_costs_example)
        business_model_1 = st.text_area('                          ')

        # Вопрос: Какие каналы вы используете для привлечения клиентов?
        # 
        acquisition_example = pa.bis_model_acquisition_channels(value_and_description_gpt)
        st.markdown(acquisition_example)
        business_model_2 = st.text_area('                           ')

        # Вопрос: Как вы видите идите масштабирование своей бизнес-модели?
        # 
        bis_model_scaling_example = pa.bis_model_scaling(value_and_description_gpt)
        st.markdown(bis_model_scaling_example)
        business_model_3 = st.text_area('                         ')



    st.title('Трекшен и финансы')
    year_earnings = st.number_input('Годовая выручка (млн. руб.): ')
    clients_amount = st.number_input('Количество клиентов: ')
    # IF FALSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if False:
        metrics = pa.fin_metrics(value_and_description_gpt,
                                    year_earnings,
                                    clients_amount)
            
    metrics = '''
    1. APRU = ...
    
    2. Churn Rate = ...
    
    3. LT = ...
    
    4. LTV = ...
    '''
    st.markdown(metrics)

    traction_and_partners_example = 'PLACEHOLDER'

    # IF FALSE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    if False:
        traction_and_partners_example = pa.traction_and_partners(value_and_description_gpt)
    
    st.markdown(traction_and_partners_example)
    traction_and_partners = st.text_area('    ')


    st.title('Команда и борд')
    if False:
        team_gpt_example = pa.team_board(value_and_description_gpt)
        st.markdown(team_gpt_example)
    team = st.text_area('    ')

    st.title('Инвестиционный раунд')
    if False:
        investment_round_example = pa.investment_round(value_and_description_gpt)
        st.markdown(investment_round_example)
    investment_round = st.text_area('     ')

    st.title('Дорожная карта')
    roadmap = st.text_area('      ')
    if False:
        roadmap_gpt = pa.roadmap(roadmap)
        st.markdown(roadmap_gpt)

    st.title('Контакты')
    contacts = st.text_area('       ')
    if False:
        contacts_gpt = pa.contaсts(contacts)

    all_is_ready = st.button("Создать неповторимый Pitch-Deck", type='primary')
    if all_is_ready:
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
