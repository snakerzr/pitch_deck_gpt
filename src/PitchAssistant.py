import os
import json

from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory, ChatMessageHistory

from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import OpenAIFunctionsAgent
from langchain.agents import AgentExecutor

from langchain.tools import DuckDuckGoSearchRun


_ = load_dotenv(find_dotenv())  # загружаем open_ai_key


class PitchAssistant:
    """ """

    gpt3_4k: str = "gpt-3.5-turbo"
    gpt3_16k: str = "gpt-3.5-turbo-16k"
    gpt4_8k: str = "gpt-4"
    gpt4_32k: str = "gpt-4-32k"
    default_temp: str = 0.0
    high_temp: str = 0.9
    verbose: bool = False

    def __init__(self):
        """ """
        pass

    # Instantiate GPT models

    def _init_gpt3_model(self, high_temp=False) -> ChatOpenAI:
        """ """
        if high_temp:
            temperature = self.high_temp
        else:
            temperature = self.default_temp

        llm = ChatOpenAI(temperature=temperature, model=self.gpt3_4k)
        return llm

    def _init_gpt3_16k_model(self, high_temp=False) -> ChatOpenAI:
        """ """
        if high_temp:
            temperature = self.high_temp
        else:
            temperature = self.default_temp

        llm = ChatOpenAI(temperature=temperature, model=self.gpt3_16k)
        return llm

    def _init_gpt4_model(self, high_temp=False) -> ChatOpenAI:
        """ """
        if high_temp:
            temperature = self.high_temp
        else:
            temperature = self.default_temp

        llm = ChatOpenAI(temperature=temperature, model=self.gpt4_8k)
        return llm

    def _init_gpt4_32k_model(self, high_temp=False) -> ChatOpenAI:
        """ """
        if high_temp:
            temperature = self.high_temp
        else:
            temperature = self.default_temp

        llm = ChatOpenAI(temperature=temperature, model=self.gpt4_32k)
        return llm

    # Create chains

    def _create_llmchain(
        self,
        prompt: str,
        llm: ChatOpenAI,
    ) -> LLMChain:
        """ """
        template = ChatPromptTemplate.from_template(prompt)
        chain = LLMChain(prompt=template, llm=llm, verbose=self.verbose)

        return chain

    # Create agents

    def _create_ddg_agent(
        self,
        system_message_prompt: str,
        llm: ChatOpenAI,
    ) -> AgentExecutor:
        """

        system_message_prompt:
            Example:
            'You are very powerful assistant,
            that creates a TOM, SAM, SOM analysis.'
        """

        tools = [DuckDuckGoSearchRun()]
        system_message = SystemMessage(content=system_message_prompt)
        prompt = OpenAIFunctionsAgent.create_prompt(system_message=system_message)
        agent = OpenAIFunctionsAgent(llm=llm, tools=tools, prompt=prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=self.verbose)

        return agent_executor

    def _create_ddg_chat_zero_shot_agent(self, llm: ChatOpenAI):
        tools = [DuckDuckGoSearchRun()]
        agent = initialize_agent(
            tools,
            llm,
            agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
        )
        return agent

    # Startup name and short description

    def short_description(self, text: str) -> str:
        prompt_question = """Тебе нужно кратко 5 словами описание проекта из {text}, который написал стартапер. У тебя есть инструкция, которая рассказывает, как это сделать хорошо:
        
        Пример: 
        Мы компания для трудойстройства студентов.
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(prompt_question, llm)
        result = chain(text)["text"]
        return result

    def problem(self, text: str) -> str:
        role = """Ты превосходный специалист по созданию питч деков. Сейчас ты помогаешь владельцу стартапа написать отличный питч дек."""

        details = """
        Первым этапом проработки питч-дека является проработка проблемы.
        Проблема - это то, что пытается решить стартап. 
        Проблема – это начальные условия, определяющие контекст, в котором ваша компания может расти быстро. Хорошая проблема популярна, растет, требует срочного решения, дорогостояща, обязательна и встречается часто. Инсайт в проблему показывает, почему ваше решение имеет конкурентное преимущество в плане роста. Выбирайте проблемы, которые соответствуют этим характеристикам, чтобы убедить инвесторов и привлечь внимание пользователей.

        Пример идеальных характеристик проблем для стартапа:
        Миллионы пользователей сталкиваются с этой проблемой.
        Рынок с проблемой растет на 20% ежегодно.
        Проблему пытаются решить сразу.
        Решение проблемы потребует больших затрат, что открывает возможности для высокой оплаты.
        Законодательство изменилось, создавая дополнительные проблемы для решения.
        Проблему необходимо решать несколько раз в день.
        """

        instruction = """
        Задачи:
        Кратко опиши проблему по пунктам, чтобы в каждом слове было максимум 5 слов, для слайда презентации, чтобы не было много текста. 

        Пример вывода: 
        <<- Недостаток средств для реализации
        - Мало времени для создания презентации>>
        """

        text_entry = """
        Текст стартапера:
        {text}
        """

        conversation_prompt = role + details + instruction + text_entry

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(conversation_prompt, llm)
        result = chain(text)["text"]

        return result

    def value(self, text: str) -> str:
        """ """

        question_prompt = """Напиши ценностное предложение стартапа из ответа {text} по пунктам.

        Учти след. информацию:
        <<Если вы строите маркетплейс или платформу, которая может монополизировать рынок, это может быть значительным преимуществом. Важно также удостовериться, что вы сможете укрепить свои позиции с ростом.>>

        
        Пример вывода: <<Everytalent - платформа для подбора персонала на базе искусственного интеллекта.
        ● Подбирает молодых талантов в компании и на рабочие места на основе результатов их оценки
        ● Мотивирует соискателей рекомендациями по превращению слабых сторон в сильные
        ● Оценивает и определяет навыки и компетентность лиц, ищущих работу>>


        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(question_prompt, llm)
        raw_result = chain(text)["text"]

        gpt_prompt = """У тебя есть текст {text}, по которому будут рассказывать презентацию, тебе нужно сократить этот текст до минимума, чтобы поместить на слайд презентации. Дополнительно есть инструкции к формированию:


        !!<<Фокус на суть
        Использование ключевых слов
        Акцент на преимуществах
        Эмоциональная привлекательность
        Избегайте повторений
        Сжатость
        Зрительная привлекательность>>!!


        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(gpt_prompt, llm)
        result = chain(raw_result)["text"]

        return result

    def solution(self, text: str) -> str:
        """ """

        solution_prompt = """
        Задачи:
        Кратко опиши решение по пунктам, чтобы в каждом слове было максимум 5 слов, для слайда презентации, чтобы не было много текста по описанию текста {text} 

        !!Каждый пункт должен содержать максимум 5 слов!!

        Пример вывода: 
        !<<- Создание бесплатного курса
        - Создание чат-бота для поиска вакансий>>!
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(solution_prompt, llm)
        result = chain(text)["text"]

        return result

    def market_tam(self, text: str, place="russia") -> str:
        """ """

        tom_sys_prompt = "You are very powerful assistant, that creates a TOM analysis."

        # llm = self._init_gpt3_model()
        llm = self._init_gpt4_model()
        agent = self._create_ddg_agent(tom_sys_prompt, llm)

        # if international:
        #     place = 'the world'
        # else:
        #     place = 'russia'

        prompt = f"Find Market Research for a {text} in {place}. And give me a number either in rubbles or in amount of clients or demand."

        result = agent(prompt)["output"]

        return result

    def market_tam_summary(self, text: str) -> str:
        """ """

        market_tom_summary_prompt = """Extract only
        market size:
        number of clients:
        demand: 

        If there is no info write: na

        Text to extract from:
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(market_tom_summary_prompt, llm)
        result = chain(text)["text"]

        return result

    def market_sam_instruction(self, text: str) -> str:
        """ """

        sam_instruction_for_client_prompt = """

        Измени шаблон инструкции в соответствии с описанием компании, чтобы владельцу компании было легко собрать необходимые данные и посчитать SAM.

        Шаблон инструкции:
        ```
        Шаг 0: Берем TAM в виде денежного объема и/или кол-ва клиентов и/или объема спроса.

        Шаг 1: Разделение Потребности по Типам Прямой и Косвенной Конкуренции

        Идентифицируйте различные типы конкуренции на рынке - прямую (конкуренты, предоставляющие аналогичные продукты или услуги) и косвенную (альтернативные решения для удовлетворения потребности).
        Разделите объем потребности из TAM между этими типами конкуренции, чтобы понять, какие части рынка контролируют ваши конкуренты, а какие - открыты для вас.

        Шаг 2: Допущение о Доли Рынка

        Проанализируйте, какую долю рынка вы хотели бы получить у конкурентов. Это может быть основано на вашей стратегии роста и конкурентных преимуществах.

        Шаг 3: Допущение о Новых Клиентах

        Рассмотрите возможность привлечения новых клиентов, которые ранее не пользовались аналогичными продуктами или услугами. Сделайте предположение о том, сколько из этих клиентов могли бы стать вашими клиентами благодаря вашим преимуществам (например, более низкая цена, улучшенные характеристики продукта).

        Шаг 4: Рассчет Количества Клиентов и Потребностей

        Используя знания о количестве потребности на потребителя (полученные при расчете TAM) и допущения из предыдущих шагов, рассчитайте количество клиентов и потребностей, которые вы можете охватить в рамках вашей стратегии.
        ```

        Описание компании:
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(sam_instruction_for_client_prompt, llm)
        result = chain(text)["text"]

        return result

    def market_som_instruction(self, text: str) -> str:
        """ """

        som_instruction_for_client_prompt = """

        Измени шаблон инструкции в соответствии с описанием компании, чтобы владельцу компании было легко собрать необходимые данные и посчитать SAM.

        Шаблон инструкции:
        ```
        **Шаг 1: Берем объем клиентов и потребностей из SAM**
        Изучите Serviceable Addressable Market (SAM), который вы предварительно определили. Это количество клиентов, которых вы можете реалистично обслужить, учитывая ваши ресурсы и ограничения.

        **Шаг 2: Определяем**
        1. Ресурсы необходимые для привлечения потребителей и обеспечения сделок. Оцените, сколько средств, персонала и времени потребуется для маркетинга, продаж и обслуживания клиентов.
        2. Последовательно определяем факторы ограничений, которые снижают потенциал компании на доступ к рассчитанному объему клиентов. Это может включать в себя конкурентов, ограничения ресурсов и другие факторы, которые могут ограничить вашу способность привлечь клиентов.
        3. Пытаемся учесть изменчивость этих ресурсов за выбранный период оценки рынка. Учтите, что ресурсы и условия могут меняться, поэтому стоит предусмотреть эту изменчивость в оценке.

        **Шаг 3: Делаем допущение о среднем чеке одной сделки**
        Оцените, какую сумму в среднем приносит одна сделка или клиент. Это может быть вашим средним доходом от продажи продукта или услуги.

        **Шаг 4: Рассчитываем итоговый объем клиентов и доходов**
        Используя данные из Шага 2, определите, сколько клиентов компания реально может обслуживать с учетом ресурсов и ограничений.
        1. Умножьте количество клиентов, которых компания реально может обработать (из Шага 2) на ваше допущение о среднем чеке сделки. Это даст вам оценку доходов от тех клиентов, которых вы фактически сможете обслуживать.
        2. Учтите факторы изменчивости и ограничения, которые вы установили на Шаге 2, чтобы сделать оценку более точной.
        ```

        Описание компании:
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(som_instruction_for_client_prompt, llm)
        result = chain(text)["text"]

        return result

    def competitors_list(self, short_company_description: str) -> str:
        """ """

        agent_description = """You are very powerful assistant, that finds competitors in the internet. If your search doesnt yeild result,
        you can be creative and try different creative search queries for up to 3 times."""

        prompt = (
            f"Find competitors for a company that {short_company_description} in russia"
        )

        # llm = self._init_gpt3_model()
        # llm = self._init_gpt4_model(high_temp=True)
        llm = self._init_gpt4_model(high_temp=False)
        agent = self._create_ddg_agent(agent_description, llm)
        raw_result = agent(prompt)["output"]

        # shorten for a slide
        competitor_prompt = """
        Задачи:
        Напиши конкурентов по пунктам, для слайда презентации, чтобы не было много текста по описанию текста {text} 

        !!Используй исключительно название компании!!
        !!Если нет компаний, ничего не пиши!!

        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(competitor_prompt, llm)
        result = chain(raw_result)["text"]

        return result

    def competitive_advantages(
        self, company_description: str, competitors_description: str
    ) -> str:
        """ """

        agent_description = """You are a very powerful assistant who determines the competitive advantages of the company in relation to its competitors. You are given the name of the company and its description, as well as the names of competitors. If necessary, you can search for additional information about competitors in order to deduce the company's competitive advantages."""

        # llm = self._init_gpt3_model()
        # llm = self._init_gpt4_model(high_temp=True)
        llm = self._init_gpt4_model(high_temp=False)
        agent = self._create_ddg_agent(agent_description, llm)

        text = f"""Название и описание компании: {company_description}

        Текст, в котором присутствуют конкуренты:
        {competitors_description}

        Тебе нужно вывести конкурентные преимущества компании.
        """
        raw_result = agent(text)["output"]

        # shorten
        advantages_prompt = """
        Задача:
        Напиши преимущества по пунктам для слайда презентации, используя максимум 5 слов для пункта, чтобы не было много текста по описанию текста {text}. 

        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(advantages_prompt, llm)
        result = chain(raw_result)["text"]

        return result

    def bis_model_profit_generation_and_costs(self, company_description: str) -> str:
        """ """

        prompt = """Напиши способы генерация дохода и основные затраты бизнеса по описанию компании.

        Описание компании:
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(prompt, llm)
        raw_result = chain(company_description)["text"]

        costs_prompt = '''
        Задача:
        Напиши текст сокращенно по пунктам для слайда презентации, используя максимум 5 слов для пункта, чтобы не было много текста по описанию текста {text}. 

        '''

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(costs_prompt, llm)
        result = chain(raw_result)["text"]

        return result

    def bis_model_acquisition_channels(self, company_description: str) -> str:
        """ """

        prompt = """Напиши о том, какие каналы привлечения клиентов используются в бизнесе по описанию компании.

        Описание компании:
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(prompt, llm)
        raw_result = chain(company_description)["text"]

        channels_prompt = '''
        Задача:
        Напиши текст сокращенно по пунктам для слайда презентации, используя максимум 5 слов для пункта, чтобы не было много текста по описанию текста {text}. 

        '''

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(channels_prompt, llm)
        result = chain(raw_result)["text"]

        return result

    def bis_model_scaling(self, company_description: str) -> str:
        """ """

        prompt = """Напиши о том, как можно масштабировать бизнес модель по описанию компании.

        Описание компании:
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(prompt, llm)
        raw_result = chain(company_description)["text"]

        scaling_prompt = '''
        Задача:
        Напиши текст сокращенно по пунктам для слайда презентации, используя максимум 5 слов для пункта, чтобы не было много текста по описанию текста {text}. 
        '''

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(scaling_prompt, llm)
        result = chain(raw_result)["text"]

        return result

    def fin_metrics(
        self, company_description: str, revenue: int, client_n: int
    ) -> dict:
        """ """

        classification_prompt = """Сообщение нужно отнести только к определенному классу:
        <<Тебе нужно вывести только число!>>

        0 - Бизнес относится к "Энергетика и коммунальные услуги"
        1 - Бизнес относится к "ИТ-услуги"
        2 - Бизнес относится к "ПО"
        3 - Бизнес относится к "Производство промышленного оборудования"
        4 - Бизнес относится к "Финансовый сектор услуг"
        5 - Бизнес относится к "Профессиональные услуги: бухгалтерские, юридические, инженерные и т.д.""
        6 - Бизнес относится к "Телекоммуникации"
        7 - Бизнес относится к "Производство товаров"
        8 - Бизнес относится к "Транспорт и перевозки"
        9 - Бизнес относится к "Производство потребительских товаров"
        10 - Бизнес относится к "Оптовая торговля"

        Пример вывода: 1

        СООБЩЕНИЕ:<<<{message}>>>
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(classification_prompt, llm)
        bis_class = chain(company_description)["text"]

        churn_rate_by_industry = {
            "Энергетика и коммунальные услуги": 0.11,
            "ИТ-услуги": 0.12,
            "ПО": 0.14,
            "Производство промышленного оборудования": 0.17,
            "Финансовый сектор услуг": 0.19,
            "Профессиональные услуги: бухгалтерские, юридические, инженерные и т.д.": 0.27,
            "Телекоммуникации": 0.31,
            "Производство товаров": 0.35,
            "Транспорт и перевозки": 0.40,
            "Производство потребительских товаров": 0.40,
            "Оптовая торговля": 0.56,
        }

        revenue = 100000
        client_n = 300
        arpu = revenue / client_n
        churn_rate = list(churn_rate_by_industry.values())[int(bis_class)]
        lt = 1 / churn_rate
        ltv = arpu * lt

        result = {"ARPU": arpu, "churn_rate": churn_rate, "LT": lt, "LTV": ltv}

        return result

    def traction_and_partners(self, company_description: str) -> str:
        """ """

        prompt = """Напиши пример трекшна и партнерствах в бизнесе по описанию компании.

        Термин "трекшн" (англ. "traction") в бизнес-контексте обозначает показатель или показатели, которые демонстрируют рост, прогресс или успешность стартапа или бизнеса. Это может быть количественные данные, которые отражают увеличение клиентской базы, продажи продуктов или услуг, пользовательскую активность, доходы и т.д. Короче говоря, трекшн - это подтверждение того, что бизнес-идея работает на практике и привлекает интерес со стороны пользователей или клиентов.

        Когда стартап или компания ищет инвестиции или партнерство, трекшн играет важную роль. Инвесторы и потенциальные партнеры обращают внимание на трекшн, чтобы оценить, насколько успешным и перспективным является бизнес. Увеличение трекшн с течением времени указывает на то, что бизнес-модель работает, а клиенты или пользователи заинтересованы в продукте или услуге.

        Примеры показателей трекшн включают в себя:

        Количество активных пользователей или клиентов.
        Объем продаж продуктов или услуг.
        Рост доходов.
        Количество скачиваний (для мобильных приложений).
        Количество подписчиков или подписок (для подписочных сервисов).
        Уровень удержания клиентов (retention rate).
        Метрики вовлеченности пользователей (например, время, проведенное на платформе).
        Общий смысл трекшн заключается в демонстрации того, что бизнес растет и привлекает внимание аудитории.

        Описание компании:
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(prompt, llm)
        result = chain(company_description)["text"]

        return result

    def team_board(self, company_description: str) -> str:
        """ """

        question_prompt = """Напиши короткую информацию о команде, бекграунде и текущих инвесторах/эдвайзерах {text} по пунктам.


        Пример: 


        Вывод : <<.
        ● Пункт 1
        ● Пункт 2
        ● Пункт 3>>


        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(question_prompt, llm)
        result = chain(company_description)["text"]

        return result

    def investment_round(self, company_description: str) -> str:
        """ """

        question_prompt = """Сделай описание инвестиционного раунда для компании:
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(question_prompt, llm)
        result = chain(company_description)["text"]

        return result

    def roadmap(self, text: str) -> str:
        """ """

        question_prompt = """Напиши дорожную карту по описанию {text} по пунктам. Дорожная карта должна представлять себя год и что в этот год будет сделано.

        Если человек указал года, то указывай их так, как указал человек.


        Пример: 


        Вывод : <<Everytalent - платформа для подбора персонала на базе искусственного интеллекта.
        ● Подбирает молодых талантов в компании и на рабочие места на основе результатов их оценки
        ● Мотивирует соискателей рекомендациями по превращению слабых сторон в сильные
        ● Оценивает и определяет навыки и компетентность лиц, ищущих работу>>


        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(question_prompt, llm)
        raw_result = chain(text)["text"]

        gpt_prompt = """У тебя есть текст {text}, тебе нужно сократить этот текст до минимума, чтобы поместить на слайд презентации. Дополнительно есть инструкции к формированию:

        !!Не используй текст из инструкций в тексте, который пишешь!!


        !!<<Фокус на суть
        Использование ключевых слов
        Акцент на преимуществах
        Эмоциональная привлекательность
        Избегайте повторений
        Сжатость
        Зрительная привлекательность>>!!


        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(gpt_prompt, llm)
        result = chain(raw_result)["text"]

        return result

    def contaсts(self, text: str) -> str:
        """ """

        prompt = """Сформируй текст для слайда из текста:
        
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(prompt, llm)
        result = chain(text)["text"]

        return result
    
    def _text_preproc_for_presentation(self,streamlit_session_state:dict) -> str:
        '''
        '''

        company_name = streamlit_session_state['company_name']
        short_description  = streamlit_session_state['short_description']
        short_description_gpt  = streamlit_session_state['short_description_gpt']
        problem_description  = streamlit_session_state['problem_description']
        problem_description_gpt  = streamlit_session_state['problem_description_gpt']
        value_and_description  = streamlit_session_state['value_and_description']
        value_and_description_gpt  = streamlit_session_state['value_and_description_gpt']
        solution  = streamlit_session_state['solution']
        solution_gpt  = streamlit_session_state['solution_gpt']
        tam  = streamlit_session_state['tam']
        tam_summary  = streamlit_session_state['tam_summary']
        sam_instruction  = streamlit_session_state['sam_instruction']
        sam  = streamlit_session_state['sam']
        som_instruction  = streamlit_session_state['som_instruction']
        som  = streamlit_session_state['som']
        competitors_list_gpt  = streamlit_session_state['competitors_list_gpt']
        competitive_advantages_gpt  = streamlit_session_state['competitive_advantages_gpt']
        profit_and_costs_example  = streamlit_session_state['profit_and_costs_example']
        business_model_1  = streamlit_session_state['business_model_1']
        acquisition_example  = streamlit_session_state['acquisition_example']
        business_model_2  = streamlit_session_state['business_model_2']
        bis_model_scaling_example  = streamlit_session_state['bis_model_scaling_example']
        business_model_3  = streamlit_session_state['business_model_3']
        year_earnings  = streamlit_session_state['year_earnings']
        clients_amount  = streamlit_session_state['clients_amount']
        metrics  = streamlit_session_state['metrics']
        traction_and_partners_example  = streamlit_session_state['traction_and_partners_example']
        traction_and_partners  = streamlit_session_state['traction_and_partners']
        team  = streamlit_session_state['team']
        team_gpt_example  = streamlit_session_state['team_gpt_example']
        investment_round_example  = streamlit_session_state['investment_round_example']
        investment_round  = streamlit_session_state['investment_round']
        roadmap  = streamlit_session_state['roadmap']
        roadmap_gpt  = streamlit_session_state['roadmap_gpt']
        contacts  = streamlit_session_state['contacts']
        contacts_gpt  = streamlit_session_state['contacts_gpt']

        result = f'''
        Для титульного слайда.
        Название компании: {company_name}
        Краткое описание по версии владельца: {short_description}
        Альтерантивное краткое описание для выбора: {short_description_gpt}

        Для слайда проблема.
        Описание проблемы по версии владельца: {problem_description}
        Альтернативное описание проблемы: {problem_description_gpt}

        Для слайда описание и ценностное предложение стартапа.
        Описание и ценностное предложение стартапа по версии владельца: {value_and_description}
        Альтернативное описание и ценностное предложение стартапа: {value_and_description_gpt}

        Для слайда решение.
        Описание решения по версии владельца: {solution}
        Альтернативное описание решения: {solution_gpt}

        Для слайда Рынок.
        Total Adressable Market для слайда Рынок: {tam}
        Численные показатели Total Adressable Market для слайда Рынок: {tam_summary}
        Serviceable Available Market для слайда Рынок: {sam}
        Serviceable Obtainable Market для слайда Рынок: {som}

        Для слайда конкуренты.
        Список и описание конкурентов: {competitors_list_gpt}
        Конкурентные преимущества:  {competitive_advantages_gpt}

        Для слайда бизнес-модель и ценообразование
        Способ генерации дохода и основные затраты по версии владельца: {business_model_1}g
        Каналы привлечения клиента по версии владельца: {business_model_2}
        Масштабирование бизнес-модели по версии владельца {business_model_3}

        Для слайда трекшн и финансы.
        Годовая выручка: {str(year_earnings)} млн. руб
        Количество клиентов: {str(clients_amount)}
        Метрики: {str(metrics)}
        Трекшн и партнеры по версии клиента: {traction_and_partners}

        Для слайда команда и борд.
        Команда и борд по версии владельца:  {team}

        Для слайда инвестиционный раунд.
        Описание инвестиционного раунда и целей инвестиций по версии владельца: {investment_round}

        Для слайда роадмап.
        Роадмап по версии владельца: {roadmap}

        Для слайда контакты.
        Контакты по версии владельца: {contacts}

        
        '''
        return result
    
    # Финальное создание текста презентации
    
    def create_presentation_text(self,streamlit_session_state:dict) -> str:
        '''
        '''
        info = self._text_preproc_for_presentation(streamlit_session_state)
        prompt = '''ИНСТРУКЦИЯ:
        Составь презентацию из 12 слайдов:
        1. Титульный - с названием компании и кратким описанием
        2. Проблема - описание проблемы, которую решает стартап
        3. Описание и ценностное предложение стартапа
        4. Решение
        5. Рынок
        6. Конкуренты
        7. Бизнес-модель и ценообразование
        8. Трекшн и финансы
        9. Команда и борд
        10. Инвестиционный раунд
        11. Роадмап
        12. Контактная информация

        {text}
        '''

        llm = self._init_gpt4_model()
        chain = self._create_llmchain(prompt,llm)
        result = chain(info)['text']

        return result
    
    def create_presentation_text_agent(self,streamlit_session_state:dict) -> str:
        '''
        '''
        info = self._text_preproc_for_presentation(streamlit_session_state)
        prompt = '''ИНСТРУКЦИЯ:
        Составь презентацию из 12 слайдов:
        1. Титульный - с названием компании и кратким описанием
        2. Проблема - описание проблемы, которую решает стартап
        3. Описание и ценностное предложение стартапа
        4. Решение
        5. Рынок
        6. Конкуренты
        7. Бизнес-модель и ценообразование
        8. Трекшн и финансы
        9. Команда и борд
        10. Инвестиционный раунд
        11. Роадмап
        12. Контактная информация

        Текст:
        ```
        {text}
        ```

        Если каких-то пунктов не хватает или их нет можешь найти информацию об этом в интернете или придумать.
        Все слайды должны быть заполнены.

        Напиши по русски.
        '''

        llm = self._init_gpt4_model(high_temp=False)
        chain = self._create_ddg_agent(prompt,llm)
        result = chain(info)['output']

        return result
