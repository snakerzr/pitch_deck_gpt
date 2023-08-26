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

    def _init_gpt4_16k_model(self, high_temp=False) -> ChatOpenAI:
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
        У проблемы, которая привлекательная для инвесторов есть характеристики:

        1. Популярность: Популярность проблемы означает, что она актуальна для большого числа людей. Большой спрос на решение этой проблемы подразумевает, что существует значительная группа людей, которые сталкиваются с ней и ищут способы ее решения. Чем более широко проблема распространена, тем выше потенциал для быстрого роста вашей компании. В сущности, это уже текущий объем потенциальных клиентов.

        2. Рост: Рост проблемы означает, что рынок, связанный с этой проблемой, растет быстрее, чем другие рынки. Это может быть обусловлено повышенным спросом на решение данной проблемы или изменениями в обществе, экономике или технологиях, которые стимулируют увеличение количества людей, сталкивающихся с этой проблемой. Иметь дело с растущей проблемой означает, что у вашей компании есть потенциал привлечь все больше пользователей и клиентов, что может способствовать быстрому росту бизнеса. В сущности, это степень роста рынка этой проблемы.

        3. Срочность: Проблема считается срочной, если ее решение требуется немедленно или в ближайшем будущем. Срочные проблемы обычно вызывают дискомфорт, требуют внимания и действий в краткосрочной перспективе. Когда пользователи осознают, что проблема требует немедленного решения, это может способствовать активному поиску подходящих решений, включая продукты и услуги, предлагаемые вашим стартапом. В сущности, это характеристика или степень безотлагательности решения проблемы.

        4. Затраты: В данном контексте затраты на решение проблемы скорее означают, что пользователи могут быть готовы платить значительные суммы за эффективное решение данной проблемы. То есть, если проблема дорогостояща для пользователей, то ваш стартап может предлагать решение, которое оправдывает эти затраты. В сущности, это денежный объем рынка проблемы.

        5. Обязательность: Проблема с обязательностью означает, что люди должны решить ее по каким-то обстоятельствам. Это может быть связано с законодательством, регулированием, требованиями индустрии или другими факторами, которые обязывают пользователей найти способы решить данную проблему. Проблемы, которые обязательны к решению, создают своеобразную неотложность для пользователей, и это может стимулировать их активное взаимодействие с вашим стартапом, предоставляющим решение этой обязательной проблемы.

        6. Частота: Проблема с частотой означает, что она проявляется у людей с высокой регулярностью или в определенные моменты времени. Это может быть проблема, с которой пользователи сталкиваются неоднократно, возможно, даже ежедневно или еженедельно. Проблемы с высокой частотой обеспечивают множество возможностей для вашего стартапа взаимодействовать с пользователями и предоставлять им решения. Когда проблема проявляется с высокой частотой, это может создать более постоянную потребность в решении, что способствует удержанию пользователей и стимулирует повторные взаимодействия с вашим продуктом или услугой.

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
        Постарайся расписать проблему, опираясь на текст стартапера, пройдясь по всем пунктам хорошей проблемы, которая заинтересовывает инвесторов.
        Сформируй содержание слайда с акцентом на те характеристики, которые являются наиболее привлекательными для инвестора.
        """

        text_entry = """
        Текст стартапера:
        {text}
        """

        conversation_prompt = role + details + instruction + text_entry

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(conversation_prompt, llm)
        raw_result = chain(text)["text"]

        # style correction

        style_description = """
        СТИЛЬ:
        ```
        Here are the detailed instructions or style of speech for each of the mentioned topics:

        Simplicity and Clarity:

        Use straightforward language that anyone can understand, avoiding complex terminology or convoluted explanations.
        Present your idea in a way that leaves no room for interpretation or confusion. Make it crystal clear.
        Focus on making your idea easily digestible, ensuring that even someone unfamiliar with your field can comprehend it.
        Eliminate any vagueness or ambiguity in your description. Be explicit about what your startup does and the problem it solves.
        Stay away from using jargon, buzzwords, or language that might not resonate with a broader audience.
        Conciseness:

        Condense your description to the most essential points. Trim away any extraneous details.
        Craft a succinct and powerful statement that encapsulates the core of your startup's value proposition.
        Prioritize brevity while still conveying the crucial aspects of your idea. Every word should serve a purpose.
        Avoiding Padding:

        Remove any unnecessary fluff or filler content from your pitch or description.
        Stay focused on the key elements that make your startup unique and valuable. Discard information that doesn't contribute to this understanding.
        Resist the temptation to add defensive explanations or preemptive responses to potential concerns. Let your idea stand on its merits.
        "X for Y" Formula:

        Choose a well-known and successful entity (X) that shares similarities with your startup's concept.
        Clearly state the comparison: "We are the X for Y," indicating the familiar model you're adapting.
        Highlight how Y, your target market or segment, can benefit from the attributes of X's model.
        Ensure that the connection between X and Y is easily understandable, even to someone unfamiliar with either.
        Empathy for Investors:

        Put yourself in the shoes of investors who have limited time to review many ideas.
        Craft your pitch and descriptions in a way that optimizes their efficiency and makes them enjoyable to read.
        Present your startup idea in a format that quickly grabs their attention and conveys its unique value.
        Conversational and Understandable:

        Imagine explaining your idea to someone who is not well-versed in your industry or field.
        Strive for a conversational tone, as if you were chatting with a friend about your startup.
        Use language that resonates with a general audience, avoiding technical jargon or complex terminology.
        Make your idea easy to remember and repeat. Aim for a description that naturally lends itself to word-of-mouth sharing.
        Empathy for Users and Customers:

        Prioritize the needs of your potential users and customers by making your idea accessible and comprehensible to them.
        Frame your pitch in a way that resonates with a broad audience, not just those deeply involved in your industry.
        Imagine someone who has never encountered your concept before and ensure that your description helps them grasp its significance.
        In summary, the style of speech involves using simple, clear, and concise language while avoiding unnecessary padding or jargon. Incorporate empathy for both investors and potential users by crafting descriptions that are easy to understand, relatable, and compelling. Utilize the "X for Y" formula effectively to highlight your startup's value proposition.
        ```
        """

        style_instruction = """
        ИНСТРУКЦИЯ:
        Примени СТИЛЬ к ТЕКСТУ.

        ТЕКСТ:
        {text}
        """

        style_prompt = style_description + style_instruction

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(style_prompt, llm)
        result = chain(raw_result)["text"]

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

        super_product_description = """
        Супер Продукт:   
        Очевидное преимущество - продукт, который значительно лучше конкурентов. Продукт должен быть на порядок (в 10 раз) более привлекательным и эффективным.
        """

        solution_instruction = """
        ЗАГЛУШКА
        {text}
        """

        solution_prompt = super_product_description + solution_instruction

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(solution_prompt, llm)
        raw_result = chain(text)["text"]

        style_description = """
        СТИЛЬ:
        ```
        Here are the detailed instructions or style of speech for each of the mentioned topics:

        Simplicity and Clarity:

        Use straightforward language that anyone can understand, avoiding complex terminology or convoluted explanations.
        Present your idea in a way that leaves no room for interpretation or confusion. Make it crystal clear.
        Focus on making your idea easily digestible, ensuring that even someone unfamiliar with your field can comprehend it.
        Eliminate any vagueness or ambiguity in your description. Be explicit about what your startup does and the problem it solves.
        Stay away from using jargon, buzzwords, or language that might not resonate with a broader audience.
        Conciseness:

        Condense your description to the most essential points. Trim away any extraneous details.
        Craft a succinct and powerful statement that encapsulates the core of your startup's value proposition.
        Prioritize brevity while still conveying the crucial aspects of your idea. Every word should serve a purpose.
        Avoiding Padding:

        Remove any unnecessary fluff or filler content from your pitch or description.
        Stay focused on the key elements that make your startup unique and valuable. Discard information that doesn't contribute to this understanding.
        Resist the temptation to add defensive explanations or preemptive responses to potential concerns. Let your idea stand on its merits.
        "X for Y" Formula:

        Choose a well-known and successful entity (X) that shares similarities with your startup's concept.
        Clearly state the comparison: "We are the X for Y," indicating the familiar model you're adapting.
        Highlight how Y, your target market or segment, can benefit from the attributes of X's model.
        Ensure that the connection between X and Y is easily understandable, even to someone unfamiliar with either.
        Empathy for Investors:

        Put yourself in the shoes of investors who have limited time to review many ideas.
        Craft your pitch and descriptions in a way that optimizes their efficiency and makes them enjoyable to read.
        Present your startup idea in a format that quickly grabs their attention and conveys its unique value.
        Conversational and Understandable:

        Imagine explaining your idea to someone who is not well-versed in your industry or field.
        Strive for a conversational tone, as if you were chatting with a friend about your startup.
        Use language that resonates with a general audience, avoiding technical jargon or complex terminology.
        Make your idea easy to remember and repeat. Aim for a description that naturally lends itself to word-of-mouth sharing.
        Empathy for Users and Customers:

        Prioritize the needs of your potential users and customers by making your idea accessible and comprehensible to them.
        Frame your pitch in a way that resonates with a broad audience, not just those deeply involved in your industry.
        Imagine someone who has never encountered your concept before and ensure that your description helps them grasp its significance.
        In summary, the style of speech involves using simple, clear, and concise language while avoiding unnecessary padding or jargon. Incorporate empathy for both investors and potential users by crafting descriptions that are easy to understand, relatable, and compelling. Utilize the "X for Y" formula effectively to highlight your startup's value proposition.
        ```
        """

        style_instruction = """
        ИНСТРУКЦИЯ:
        Примени СТИЛЬ к ТЕКСТУ.

        ТЕКСТ:
        {text}
        """

        style_prompt = style_description + style_instruction

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(style_prompt, llm)
        result = chain(raw_result)["text"]

        return result

    def market_tam(self, text: str) -> str:
        """ """

        tom_sys_prompt = (
            "You are very powerful assistant, that creates a TOM, SAM, SOM analysis."
        )

        llm = self._init_gpt3_model()
        agent = self._create_ddg_agent(tom_sys_prompt, llm)
        result = agent(text)['output']

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

        llm = self._init_gpt3_model()
        agent = self._create_ddg_agent(agent_description, llm)
        result = agent(prompt)["output"]

        return result

    def competitive_advantages(
        self, company_description: str, competitors_description: str
    ) -> str:
        """ """

        agent_description = """You are a very powerful assistant who determines the competitive advantages of the company in relation to its competitors. You are given the name of the company and its description, as well as the names of competitors. If necessary, you can search for additional information about competitors in order to deduce the company's competitive advantages."""

        llm = self._init_gpt3_model()
        agent = self._create_ddg_agent(agent_description, llm)

        text = f"""Название и описание компании: {company_description}

        Текст, в котором присутствуют конкуренты:
        {competitors_description}

        Тебе нужно вывести конкурентные преимущества компании.
        """
        result = agent(text)["output"]

        return result

    def bis_model_profit_generation_and_costs(self, company_description: str) -> str:
        """ """

        prompt = """Напиши способы генерация дохода и основные затраты бизнеса по описанию компании.

        Описание компании:
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(prompt, llm)
        result = chain(company_description)["text"]

        return result

    def bis_model_acquisition_channels(self, company_description: str) -> str:
        """ """

        prompt = """Напиши о том, какие каналы привлечения клиентов используются в бизнесе по описанию компании.

        Описание компании:
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(prompt, llm)
        result = chain(company_description)["text"]

        return result

    def bis_model_scaling(self, company_description: str) -> str:
        """ """

        prompt = """Напиши о том, как можно масштабировать бизнес модель по описанию компании.

        Описание компании:
        {text}
        """

        llm = self._init_gpt3_model()
        chain = self._create_llmchain(prompt, llm)
        result = chain(company_description)["text"]

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
