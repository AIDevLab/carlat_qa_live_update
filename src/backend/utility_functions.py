

import os, sys
from os import path
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
sys.path.append( path.dirname(path.dirname(path.abspath(__file__))))
from outlines.models.openai import OpenAIConfig
from outlines import models
from pydantic import BaseModel, constr
import outlines
import json
import re
from Levenshtein import distance
from spire.doc import *
from spire.doc.common import *
from openai import OpenAI
import json
import traceback 
import concurrent.futures
from pydantic import BaseModel

# Get the absolute path to the main directory (my_project)
main_directory = os.path.abspath(os.path.dirname(__file__))
# Add main directory and its subdirectories to sys.path
sys.path.append(main_directory)

client = OpenAI(api_key=st.secrets["api_key"])


def updates_session_text_area():
    """
    Update the current topic's quotes and formatted QA in the session state.

    This function updates the `topics_dict` in the session state with the
    current values from `quotes_text_area` and `qa_text_area` for the selected topic.

    Returns:
        None
    """
    st.session_state.topics_dict[st.session_state.topic]["quotes"] = st.session_state.quotes_text_area
    st.session_state.topics_dict[st.session_state.topic]["formated_qa"] = st.session_state.qa_text_area


def get_updated_key_topics(text):
    """
    Update session state with a list of non-empty topics from the given text.
    Splits the input text by newline characters and updates `list_topics`
    in the session state with non-empty lines.

    Args:
        text (str): The input text containing topics separated by newlines.

    Returns:
        None
    """
    st.session_state.list_topics = [item for item in text.split("\n") if item != ""]


def get_quotes_qa_ourlines(transcript, topic):
    """
    extract quotes that are related to a specific topic from a trasncript

    Args:
        transcript (str): Theinterview transcript
        topic (str): the topic that the quotes are related to

    Returns: json-like string
            {"quotes": "....",
            "questions": [...],
            "answers": [...]}
    """
    config = OpenAIConfig(
        presence_penalty=1.,
        top_p=0.00000001,
        temperature=0.00000001,
        seed=0,
    )
    model = models.openai("gpt-4o-mini", config=config)

    class Output(BaseModel):
        quotes: list
        questions: list
        answers: list

    # Construct structured sequence generator
    generator = outlines.generate.json(model, Output)
    output = generator("""Given an interview transcript and a specific topic your role is to extract all the quotes related to that topic from the topics. 
                        After that, use those quotes to generat from 1 to 3 well-worded questions- answers that cover what appeared in the extracted quotes.
                        The final output must be in a python parsable JSON format as follows:
                        
                        {"quotes": "....",
                        "questions": [...],
                        "answers": [...]}

                        Other instructions:
                        + Return all the exact quotes from the transcript that are related to the specified topic.
                        + Return the quotes as they have appeared, no changes should be made.
                        ----------------------------------------------------------------
                        Topic: """+" " + topic
                        
                        +"""
                        Start of Transcript:
                        """ 
                        +" " + transcript
                        +"""
                        End of Transcript:
                        """)
    return output


def process_key_topics(topics_string):
    """
    split topics strings into topics in a form of a list

    Args:
        topics_string (str): The topics in a one string
    Returns:
        list of topics
    """
    topics = topics_string.strip().split('\n')
    topics = [topic.strip() for topic in topics]
    return topics


def process_quotes(quotes):
    """
    format the quotes in a string
    Args:
        quotes (list): The list of quotes
    Returns:
        formated string of quotes
    """
    content = ""
    for q in quotes:
        content = content + q + "\n\n"

    return content


def get_key_topics(transcript, custom_instructions):
    """
    extract key topics from an interview transcript

    Args:
        transcript (str): Theinterview transcript
        custom_instructions (str): user's custom prompt 

    Returns: list-like string of key topics
    """
    message_text = [{"role":"system","content":
                    """You are a JSON formatter assistant. Your role is to follow the instructions and provide a JSON formatted response.
                    """,

                    "role":"user","content":
                    """
                    ----------------------------------------------------------------
                    Start of Transcript:\
                    """ 
                    +" " + transcript
                    +"""
                    End of Transcript:
                    ----------------------------------------------------------------
                    Given the above interview transcript, your role is to extract up to 8 key subtopics related to the main topic and display them as bullet points.\
                    + The extracted key topic can be less than 8 but not more than 8.\
                    + Each subtopic must be as elementary as possible such as (definition of ...., symptoms of ...., tretment of ...., ect)\
                    + Limit the number of subtopics to the top 6 to 8 most dominant subtopics.\
                    + Avoid using a hyphen as the leading symbol for topics
                    
                    The key topics must always be in the following format:\

                    Topic1: The name of the topic in a nominal senstence of a maximum of 2 lines.\
                    Topic2: The name of the topic in a nominal senstence of a maximum of 2 lines.\
                    Topic3: The name of the topic in a nominal senstence of a maximum of 2 lines.\
                    etc. \

                    Example :\

                    Topic1: Definition of gambling addiction and its distinguishing factors from normal gambling.\
                    Topic2: Prevalence rates and under-recognition of gambling disorder.\
                    Topic3: Classification of gambling addiction as an addictive disorder\
                    etc. 

                    ## Additional instructions:
 
                    """
                    +
                    custom_instructions
                    
                    }

                    ]

    completion = client.chat.completions.create(
    model="gpt-4o-mini", # model = "deployment_name" gpt-4o-mini
    messages=message_text,
    temperature=0.0001,
    max_tokens=300,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
    )

    topics_list = process_key_topics(completion.choices[0].message.content)
    topics = '\n\n'.join(topics_list)
    # remove leading and tailing white space
    topics = topics.strip()
    return topics, topics_list


def get_qa(quotes, topic, custom_prompt):
    """
    Given the topic and the associated quotes from an interview generate Q/A pairs.

    Args:
        quotes (str): The quotes from the interview
        topic (str): The topcic to generate Q/A pairs for
        custom_instructions (str): user's custom prompt 

    Returns: JSON-like string 
                    { 
                    "questions": List ,
                    "answers": List
                    } 
    """
    message_text = [{"role":"system","content":
                     """
                    ## Defining the profile and role 
                    You are a JSON formatter assistant and  an editor's assitant. 
                    Your role is to follow the instructions and provide a JSON formatted response.
                    
                    
                    """,


                    "role":"user","content":
                                """

                                ## Defining task
                                Given the topic and the associated quotes from an interview, your role is to use those quotes to generate between one and three well-worded questions and use the quotes provided verbatim  to answer the questions. Remove filler words and add connecting words as needed to make answers flow and feel conversational. The answers MUST be at least 200 words.
                                The final output must be in a python parsable JSON format as appears on the following schema:
                                ## Defining Rules:
                                + Always end the output by }
                                + Always put a comma after each JSON field
                                + The number of questions must always equal the number of answers
                                + Each generated question must have it's corresponding answer generated
                                + Keep the number of generated Q/A pairs under four
                                + use the quotes provided verbatim  to answer the questions
                                + answers MUST be around 200 word
                                + answers MUST sound conversational and be interesting and informative
                                + Each question should start with the interviewer label, example: 
                                Interviewer: ................
                                + Each answer should start with the name of the interviewee that is already mentionned in the bellow quotes,  example: 
                                Dr. Jane: ................
		
                                
                                ## Defining output format
                                { // opening curly brace 
                                "questions": List ,
                                "answers": List
                                } // closing curly brace 

                                
                    Topic : """+" " + topic.split(":")[1]
                    
                    +"""

                    <Quotes>
                    """  
                    +" " + quotes
                    +"""<End Quotes>
                    ## Defining output format
                    { // opening curly brace 
                    "questions": List ,
                    "answers": List
                    } // closing curly brace 

                    ## Additional instructions:
                    """+
                    custom_prompt
                    +"""

                    ## assistant :
                    """
                    
                    }

                    ]

    completion = client.chat.completions.create(
    model="gpt-4o-mini", # model = "deployment_name"
    messages=message_text,
    temperature=0.000001,
    max_tokens=4000,
    top_p=0.00001,
    frequency_penalty=0,
    presence_penalty=0,
    )

    return completion.choices[0].message.content  


def parse_response(response):
    """
   parse string response to json object

    Args:
        response (str)
    Returns:
        json object
    """
    parsed_data = json.loads(response)
    return parsed_data["questions"], parsed_data["answers"]


def format_qa_content(dict):
    """
    format the content of the generated Q/A pairs lists into a string that contains the questions and the answers alternativly

    Args:
        dict (dict): contains questions and answers lists
    Returns:
        formated string
    """
    content = ""
    for i in range(len(dict["questions"])):

        content += dict["questions"][i] + "\n" + dict["answers"][i] + "\n\n"

    return content


def generate_final_draft():
    """
    generate the final draft of using the generated Q/A pairs 

    Returns:
        final draft(str)
    """
    initial_draft = ""
    for topic in st.session_state.topics_dict.keys():
        initial_draft = initial_draft + st.session_state.topics_dict[topic]["formated_qa"] + "\n\n"
        message_text = [{"role":"system","content":
                     
                     """You are a helpful editor's assistant.
                    Given the initial question/answer pairs draft bellow, your task is to identify redundant question-answer pairs and then remove them. 
                    Sometimes the same question is asked diffrently multiple times in the draft, find this type of Q/A pairs and completly remove these pairs.
                    -These duplicates may either be exact replicas or closely similar in meaning. 
                    - Return the orginial draft without these duplicates(same questions/answers, diffrent wording) Q/A pairs
                    """,
      
                    "role":"user","content": 
                    """
                    Given the bellow initial set of questions and answers about diffrent topics, your task is to identify redundant question-answer pairs and then remove them. 
                    Sometimes, the same question is asked diffrently, find this type of Q/A and remove redundancy.
                    - These duplicates may either be exact replicas or closely similar in meaning. 
                    - These are pairs of questions and answers that convey similar or identical information, either in terms of wording or meaning.
                    - Return the orginial draft without these duplicate Q/A pairs
                    
                    <Start of Initial Q/A draft>:
                    """ 
                    +" " + initial_draft
                    +"""
                    <End of Initial Q/A draft>

                    - The output must be only Q/A pairs with no introductory words, tags or additional  comments before or after the pairs.

                    """
                    
                    }

                    ]

    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=message_text,
    temperature=0.1,
    max_tokens=8000,
    top_p=0.1,
    frequency_penalty=0,
    presence_penalty=0,
    )

    return completion.choices[0].message.content


def extract_redundancy(initial_draft):
    """
     identify redundant question-answer pairs and return them as a response. 

    Args:
        initial_draft (str)
    Returns:
        list-like string: redundant_pairs
    """

    message_text = [{"role":"system","content":
                     
                     """You are a helpful editor's assistant.""",
      
                    "role":"user","content": 
                    """
                    Given the initial question/answer pairs draft bellow, your task is to identify redundant question-answer pairs and return them as a response. 
                    Sometimes the same question is asked diffrently, find this type of Q/A and return it.

                    ## more context information
                    - These duplicates may either be exact replicas or closely similar in meaning. 
                    - These are pairs of questions and answers that convey similar or identical information, either in terms of wording or meaning.
                    - Ensure that for each topic, the redundant pairs list only includes additional redundant pairs and does not include the original pair.
                    - Take your time to identidy all closely similar Q/A pairs accross the entire draft bellow.
                    - the response should be in a list format where each item is a Q/A pair with including the speakers labels as they appeared in the initial draft.

                    Example of response format:
                    [("speaker x: question_text1", "speaker y: response_text1"),
                    ("speaker x: question_text2", "speaker y: response_text2"),
                    ("speaker x: question_text3", "speaker y: response_text3"),
                    ... ect
                    
                    ]

                  


                  <Start of Initial Q/A draft>""" 
                            + 
                            initial_draft
                            +"""
                  <End of Initial Q/A draft>
                    """
                    }

                    ]

    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=message_text,
    temperature=0.7,
    max_tokens=5000,
    frequency_penalty=0,
    presence_penalty=0,
    )

    return completion.choices[0].message.content


def substruct_redundancy(initial_draft, redundant_pairs):
    """
    remove redundant question-answer pairs from the draft using llm

    Args:
        initial_draft (str)
        redundant_pairs(list)
    Returns:
        updated_draft (str)
    """
    message_text = [{"role":"system","content":
                     
                     """You are a helpful editor's assistant.""",

      
                    "role":"user","content": 
                    """
                    Given the initial question/answer pairs draft bellow in addition to the bellow  list of redundant Q/A pairs  (the redanduncy draft),  your task is to remove the QA pairs that appeared in the the latter from the initial draft, leaving only those QA pairs that do not appear in the redanduncy draft then return this redundancy-free draft as response.

                    - The response should exclude all the Q/A pairs that appear in the redundant Q/A pairs JSON bellow in the form of question/answer pairs.
                    - The response must organize the redundancy-free Q/A pairs in the same order in which they appeared in the initial draft 
                    - The response must have the same format as the initial draft. (raw text  Q/A pairs, not JSON)
                    - Take your time to generate the  redundancy-free Q/A pairs draft in the same format of the initial Q/A draft.
                    
                  <Start of Initial Q/A draft>""" 
                            + 
                            initial_draft
                            +"""
                  <End of Initial Q/A draft>

                  <Start of redundant Q/A pairs to be removed from the Initial Draft>""" 
                  +
                  redundant_pairs
                  +
                  """ <End of  redundant Q/A pairs to be removed from the Initial Draft> """


                    }

                    ]

    completion = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=message_text,
    temperature=0.4,
    max_tokens=3000,
    frequency_penalty=0.2,
    presence_penalty=0,
    )
    return completion.choices[0].message.content  


def remove_redundant_pairs(initial_script, red_list):
    """
    remove redundant question-answer pairs from the draft using regular expressions

    Args:
        initial_draft (str)
        redundant_pairs(list)
    Returns:
        updated_draft (str)
    """
    for item in red_list:
      # Define the pattern to search for
      pattern = re.escape(item[0])+"\n"+re.escape(item[1])
      initial_script = re.sub(pattern, '', initial_script)
  
    initial_script = re.sub(r"\n{3,}", "\n\n", initial_script)
    return initial_script


def remove_duplicates_levenshtein(qa_pairs, threshold_ratio=0.65):
    """
    Remove duplicates from a list of question-answer pairs using Levenshtein Distance.

    Args:
    - qa_pairs (list): A list of tuples containing question-answer pairs.
    - threshold (int): The maximum allowed Levenshtein distance for considering pairs as duplicates.

    Returns:
    - list: A list of unique question-answer pairs.
    """
    unique_pairs = []
    for qa_pair in qa_pairs:
        is_duplicate = False
        for unique_pair in unique_pairs:
            # Calculate Levenshtein distance between question-answer pairs
            max_length = max(len(qa_pair), len(unique_pair))
            threshold = threshold_ratio * max_length
            
            qa_distance = distance(qa_pair, unique_pair)
            # If both question and answer distances are below the threshold, consider them duplicates
            if qa_distance <= threshold :
                is_duplicate = True
                break
        if not is_duplicate:
            unique_pairs.append(qa_pair)

    deduplicated_text = "\n\n".join(unique_pairs)

    deduplicated_text = re.sub(r"\n{3,}", "\n\n", deduplicated_text)
    return deduplicated_text


def parse_response_quotes(topics_dict):

    """
    Parse extracted quotes to formated string to display on UI

    Args:
    - topics_dict : A dictionnary of topcics/quotes

    Returns:
    - list: A list of quotes
    - str : A string of quotes
    """
    quotes_text = ""

    quotes_list = []

    for topic in topics_dict.keys():
        quotes_text = quotes_text + "*********" + topic + "*********"+"\n"

        quotes_text = quotes_text + "\n----------------------------------\n".join(topics_dict[topic]["quotes"])+"\n\n"
        quotes_list = quotes_list + topics_dict[topic]["quotes"]
    return quotes_text, quotes_list


def quotes_topic_str(quote, topics):

    quotes_text = ""
    quotes_text = quotes_text +  "<Quote> : \n"+ quote+"\n<START OF Topic Options> : \n-"+ "\n-".join(topics) +"\n<END OF Topic Options>\n\n\n"

    # print("555555555555555555555555555555555555555555555555555555555")
    # print(quotes_text)
    # print("555555555555555555555555555555555555555555555555555555555")
    return quotes_text


def format_qa_content_all(topics_dict):
    """
    Parse the generated Q/A pairs to formated string to display on UI

    Args:
    - topics_dict : A dictionnary of topcics/QA

    Returns:
    - str : A string of formated Q/A pairs
    """

    output_text = ""

    for topic in topics_dict.keys():
        output_text = output_text + "********************************" + topic + "********************************\n"
        output_text = output_text + topics_dict[topic]["formated_qa"] + "\n\n"

    return output_text


def highlight(inputFile, topics_dict):
    """
    highlight the given chunks in the input file
    Args:
    - inputFile : file name
    - phrases: list of words to highlight

    Returns:
    - str: name of the highlighted file
    """

    COLORS =   [
        Color.get_Yellow,
        Color.get_Blue,
        Color.get_Red,
        Color.get_Green,
        Color.get_Cyan,
        Color.get_Magenta,
        Color.get_Orange,
        Color.get_Pink,

    ]
    document = Document()
    # Load a Word document

    document.LoadFromFile(inputFile, FileFormat.Docx2016)

    # Find the first instance of a specific text
    color_index = 0
    for topic in topics_dict.keys():

        quotes_to_highlight = [quote.strip() for quote in topics_dict[topic]["quotes"]]
        
        for phrase in quotes_to_highlight:

            chunks = phrase.split("\n")
            chunks = [c for c in chunks if c != ""]

            for chunk in chunks:

                textSelection = document.FindString(chunk, True, True)
                if textSelection != None:

                    # Get the instance as a single text range
                    textRange = textSelection.GetAsOneRange()
                    # Highlight the text range with a color

                    try:
                        textRange.CharacterFormat.HighlightColor = COLORS[color_index]() 
                    except Exception as e:
                        print(e)

        color_index = color_index+1
    # Save the resulting document
    temp_output = "temp_highlighted.docx"
    document.SaveToFile(temp_output, FileFormat.Docx2016)
    document.Close()

    return temp_output


def get_memorible_quotes(transcript):
    """
    get the memorable quotes from a transcript using LLM

    Args:
    - transcript (str)

    Returns:
    - str : json_like string
            { // Opening curely brace
            quotes: [quote1, quote2, quote3, ect]
            } // Closing curely brace
    """
    message_text = [{"role":"system","content":
                     
                     """You are a helpful assistant. Your role is to follow the instructions and provide a JSON formatted response.""",

      
                    "role":"user","content": 
                    """

            
                  <Start of interview transcript>""" 
                            + 
                            transcript
                            +"""
                  <End of interview transcript>

                  Given the above transcript do the following:

                  <instructions>

                    - Identify actionable clinical advice or directives for clinical practice.
                    - Focus on the interviewee's specific clinical advice, experiences, and suggestions.
                    - Emphasize the interviewee's unique experiences and perspective.
                    - Rephrase the quotes to convey the core message without filler words or interview style.
                    - Each quote should be a complete thought with a clear beginning and end in a form of 2-3 lines paragraph.
                    - Avoid the quotes including the following : background information or general clinical practices, editorializing or self-promotion, short quotes lacking context and citations.
                    - The interviewee's voice should be prominent in the paraphrased quotes (use first-person perspective).
                    - Remove any mentions of research or publications.
                    - Each memorable quote should be in a form of 2-3 lines paragraph with as much details and context as possible.
                    - Always return your response in the bellow JSON format:
                    

                    { // Opening curely brace 
                    quotes: [quote1, quote2, quote3, ect]
                    } // Closing curely brace 

                    

                    - Follow the bellow quotes example style

                    < Start of an example of the expected output style for a transcript about depression disorder>

                    [
                    "I would highly recommend incorporating cognitive-behavioral techniques into your practice. They have proven to be incredibly effective for my patients in managing depressive symptoms and improving overall mental health.",

                    "One strategy that has worked well for me is using a combination of psychotherapy and pharmacotherapy to address both the psychological and biological aspects of depression. This holistic approach has yielded significant improvements in patient outcomes.",

                    "It's important to remember that every patient with depression is unique, and what works for one may not work for another. Take the time to understand their individual experiences and tailor your treatment plan accordingly.",

                    "I've found that incorporating lifestyle modifications, such as exercise and dietary changes, can greatly enhance the effectiveness of traditional treatments for depression. Collaborating with other healthcare professionals can provide comprehensive support for patients.",

                    "Don't underestimate the power of empathy and active listening. As clinicians, our compassion and understanding can make a significant difference in the therapeutic process and in the lives of our patients.",

                    "When it comes to prescribing antidepressants, always consider the potential risks and benefits. It's essential to have open and honest conversations with patients about the possible side effects and the expected timeline for seeing improvements.",

                    "Don't be afraid to incorporate evidence-based complementary therapies, such as mindfulness meditation and yoga. I've seen remarkable results in patients who engage in these practices alongside conventional treatments.",

                    "Building a strong therapeutic alliance with your patients is crucial in treating depression. Trust and open communication are key to fostering a collaborative relationship that encourages patients to actively participate in their recovery journey." ]

                    < End of the example>
                    <end of instructions>
                    
                   """
                   
                    }

                    ]

    completion = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages=message_text,
    temperature=0.5,
    max_tokens=3000,
    frequency_penalty=0,
    presence_penalty=0,
    )
    return completion.choices[0].message.content  



def get_qa_for_quote(quotes,topic, custom_prompt):

    message_text = [{"role":"system","content":
                     
                     """
                    ## Defining the profile and role 
                    You are a JSON formatter assistant and  an editor's assitant. 
                    Your role is to follow the instructions and provide a JSON formatted response.
                    
                    
                    """,


                    "role":"user","content":
                                """

                                ## Defining task 

                                Below is a quote from an interview transcript. Your task is to craft a question based on this quote and use the quote itself to generate a detailed answer. The answer should mirror the interviewer's wording, style, and tone precisely.
                                Ensure that the answer is not a condensed version or summary of the original quote; instead, enhance the quote by removing unnecessary words, repetitions, etc., to make it suitable for publication in a journal. Preserve the original number of paragraphs present in the quote
                                The final output must be in a python parsable JSON format as appeares on the bellow schema.


                                ## Defining Rules:
                                + Always end the output by }
                                + Always put a comma , after each JSON field 
                                + Make the answer to the question as detailed as possible according to the quote.
                                + Remove all the filler words from the quote 
                                + Relove the secondary auxilary thoughts with no direct relation to the question.
                                + Remove repetitions of thoughts from the quote
                                + make the quote-based answer more suitable for a journal article than the quote itself.
                                + Maintain the main sentences, style and wording of the quotes.
                                + Maintain the approximate length of the  quote. 
                                + Do not summarize the answer nor shrink it to less than two thirds of original quote.
                                + The question should start with the interviewer label, example: 
                                Interviewer: ................
                                + The answer should start with the name of the interviewee that is already mentionned in the bellow quotes,  example: 
                                Dr. Jane: ................
		
                                
                                ## Defining output format
                                { // opening curly brace 
                                "question": "question content" ,
                                "answer": "answer content"
                                } // closing curly brace 

                            
                    Topic """+" " + topic
                    
                    +"""

                    <Quotes>
                    """  
                    +" " + quotes
                    +"""<End Quotes>

                    ## Defining JSON output format 
                    { // opening curly brace 
                    "question": "question content" ,
                    "answer": "answer content"
                    } // closing curly brace 

                    ##Other instructions:
                    """
                        +
                        custom_prompt
                    
                    }

                    ]

    completion = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages = message_text,
    temperature=0.1,
    max_tokens=1000,
    top_p=0.01,
    frequency_penalty=0,
    presence_penalty=0,
    response_format={ "type": "json_object" }
    )

    return completion.choices[0].message.content





def find_redundant_quotes(quotes_dict):

    """"
    This  function finds the quotes that were assigned to more than one topic

    Args:
        - quotes_dict: dictionnary with topics as keys and quotes as values
    Returns:
        - a dictionary with quotes as keys and topics as values
    
    """

    duplicates = {}
    quotes = []
    # get all the quotes from the dictionary
    topics = quotes_dict.keys() 
    for topic in topics:
        quotes = quotes + quotes_dict[topic]["quotes"]

    # cast the list of quotes into a set to remove duplicates 
    quotes = set(quotes)

    # iterate on the quotes, and find, for the each quote, the topics that it was assigned to 
    for quote in quotes:
        # initialize the dictionnary with the quotes
        duplicates[quote] = []

        for topic in topics:
            if quote in quotes_dict[topic]["quotes"]:
                duplicates[quote].append(topic)


    # remove the keys with exactly one topic  (no redunduncy)


    duplicate_keys = duplicates.keys()
    to_be_del = []

    for key in duplicate_keys:
        if len(duplicates[key]) == 1:
            to_be_del.append(key)

    for k in to_be_del:
        del duplicates[k]

    return duplicates


def correct_quotes_topic_assignment(quote, topics):


    """
    This function takes the dictionnary of the redundant quotes and returns corrected quote to topic assignment

    Args:
    - redundant_quotes_dict: dictionnary of quotes as keys and topics as values

    Returns:
    - updated_topic_assignment_dict = The original dictionnary of topics with redunduncy without redunduncy
    """




    messages = [

        {
            "role": "system",
            "content": "You are a helpfull assistant and JSON formater that helps editors assign quotes to the most appropriate topic option."
        },

        {
            "role": "user",
            "content":
            """

            <START of quote-topics dictionnary>
            """ +
            quotes_topic_str(quote, topics)
            + """
            <END of quote-topics dictionnary >



            Given the above quote and a set of topic options do the following:


            <INSTRUCTIONS START>

            1. Identify up to the top two topics that the quote relates to the most from the list of options.
            2. Return an updated dictionnary in such a way that each quote(key) has up to two values(two topic, which is the one the quote relates to the most)
            3. Return the quotes exactly as they were inputed, do not modify the quotes 
            4. The selected topic should be returned in the same way it was inputed (include topic number detail)
            5. Limit the topic selection to the list of topics attributed to each quote.
            6. Do not attribute to a quote a topic that doesnt exist in its related list of topics.
	    7. Ensure that every topic has at least one quote.
            8. Do not start the topic by - 
            - Ensure the generated content is returned in a valid JSON format as illustarted bellow:

                <OUTPUT FORMAT START>

                { // Opening curly brace

                <quote text> : <topic>
               
                } // closing curly brace 
                <OUTPUT FORMAT END>

            - Ensure the generated content is returned in a valid JSON format
            <INSTRUCTIONS END>

            Follow the bellow example instance: 

            
            <START OF TOPIC SEELCTION EXAMPLE>
            <Quote> :
                Dr. Fong: This is a question I get essentially, four or five times a week, when people will say, do I have a gambling addiction? Or how do I know if my husband or wife or son or daughter is going to develop a gambling addiction? And I usually start with the following. First, I tell people gambling is part of something we do every day in our lives. It’s risk-taking. It’s decision-making. It’s going for rewards.

                Essentially, though, the difference is that between someone who gambles regularly and socially versus someone with a (quote) gambling disorder, or a gambling addiction is that if their gambling continues to bring harmful consequences to their lives and they continue to engage in gambling, that’s an addiction. Furthermore, men and women with gambling disorder, they experience all sorts of things that people who gamble recreationally do not. They have urges and cravings that get in the way of them 
                completing their daily lives. They have restrictions and limitations [00:05:00] on what they can do in life because of the consequences of gambling.

                So, much like any other addictive disorder, it isn’t so much how much you gamble or how often or how much you’ve lost, it’s what are the consequences and what are the 
                biological and psychological and social experiences it is for the person who is gambling? If they gamble in a way that’s harmful and distressing and emotionally painful, that’s an addiction. If they gamble and they have a lot of fun and they lose a lot of money, but it doesn’t impact their daily functioning, that’s not an addiction. That’s just a hobby.
                <Topic Options For above quote> :
                -Topic1: Definition of gambling addiction and its distinguishing factors from normal gambling.
                -Topic3: Classification of gambling addiction as an addictive disorder.


            LLM OUTPUT:

            {
                "Dr. Fong: This is a question I get essentially, four or five times a week, when people will say, do I have a gambling addiction? Or how do I know if my husband or wife or son or daughter is going to develop a gambling addiction? And I usually start with the following. First, I tell people gambling is part of something we do every day in our lives. It’s risk-taking. It’s decision-making. It’s going for rewards.

                Essentially, though, the difference is that between someone who gambles regularly and socially versus someone with a (quote) gambling disorder, or a gambling addiction is that if their gambling continues to bring harmful consequences to their lives and they continue to engage in gambling, that’s an addiction. Furthermore, men and women with gambling disorder, they experience all sorts of things that people who gamble recreationally do not. They have urges and cravings that get in the way of them 
                completing their daily lives. They have restrictions and limitations [00:05:00] on what they can do in life because of the consequences of gambling.

                So, much like any other addictive disorder, it isn’t so much how much you gamble or how often or how much you’ve lost, it’s what are the consequences and what are the 
                biological and psychological and social experiences it is for the person who is gambling? If they gamble in a way that’s harmful and distressing and emotionally painful, that’s an addiction. If they gamble and they have a lot of fun and they lose a lot of money, but it doesn’t impact their daily functioning, that’s not an addiction. That’s just a hobby."

                :
                "Topic1: Definition of gambling addiction and its distinguishing factors from normal gambling."
            }


            <END OF TOPIC SEELCTION EXAMPLE>
            """
        }
    ]


    completion = client.chat.completions.create(

        model = "gpt-4o-mini",
        temperature=0.4,
        max_tokens= 2000,
        frequency_penalty= 0,
        presence_penalty= 0,
        messages=messages,
        response_format={"type": "json_object"}
    )

    try:

        corrected_assignment = json.loads(completion.choices[0].message.content , strict=False)
    except Exception as e:
        print(e)
        corrected_assignment = {}
        return corrected_assignment

    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    # print("IN ")
    # print(quote[:40])
    # print("Out ")
    # print(list(corrected_assignment.keys())[0][:40])
    # print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    return corrected_assignment



def update_topic_assignment(redundant_quotes_dict, topics_dict):
    """
    This function takes the dictionnary of the redundant quotes and returns corrected quote to topic assignment

    Args:
    - redundant_quotes_dict: dictionnary of quotes as keys and topics as values
    - topics_dict: The original dictionnary of topics with redunduncy

    Returns:
    - updated_topic_assignment_dict = The original dictionnary of topics  without redunduncy
    """
    correct_topic_assign_dict = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit the processing of each quote for execution
        future_to_quote = {executor.submit(correct_quotes_topic_assignment, quote, redundant_quotes_dict[quote]): quote 
                        for quote in redundant_quotes_dict.keys()}
        
        # Wait for all futures to complete (blocks until all threads finish)
        concurrent.futures.wait(future_to_quote.keys())
        
        for future in future_to_quote.keys():
            # print("complited ... ")
            quote = future_to_quote[future]
            try:
                corrected_assign = future.result()

                key = list(corrected_assign.keys())[0]  # Get the first key
                correct_topic_assign_dict[key] = corrected_assign[key]
            except Exception as exc:
                print(f"Processing exception for quote {quote}: {exc}")

    try:
        # print("000000000000000000000000000000000000000000000000000000")
        for key in correct_topic_assign_dict.keys():
            pass
            #print(key)
            #print("\n\n-----------------------------------------------------")


        # print(len(correct_topic_assign_dict.keys()))
        # print(len(redundant_quotes_dict.keys()))
        # print("000000000000000000000000000000000000000000000000000000")
        # use the corrected assigement to modify the original topic-quotes dictionary
        for quote in redundant_quotes_dict.keys():
            # print("Entering the for loop ..........")
            #print(quote)
            try:
                pattern = re.escape(quote[:20])

                # Find the topic using re
                for q in correct_topic_assign_dict.keys():
                    if re.match(pattern, q[:20]) != None:
                        quote_ = q
                        break

                topic_to_remove_qoute_from =  list(set(redundant_quotes_dict[quote]) - set([correct_topic_assign_dict[quote_]]))
                # print("/////////////////////////////////////////////")
                # print("prev assing")
                # print(redundant_quotes_dict[quote])
                # print("new assign")
                # print(correct_topic_assign_dict[quote_])
                # print("topic_to_remove_qoute_from")
                # print(topic_to_remove_qoute_from)
                # print(correct_topic_assign_dict[quote_])
                # print(quote[:20])
                # print("-----------")
                # print(quote_[:20])
                # print("-----------")
                # print((set(redundant_quotes_dict[quote])))
                # print("-----------")
                # print(set([correct_topic_assign_dict[quote_]]))
                # print("------------")
                # print(topic_to_remove_qoute_from)
                # print("/////////////////////////////////////////////")
                      
                    
                # print("redundant_quotes_dict")
                # print(redundant_quotes_dict[quote])
                # print("topic_to_remove_qoute_from")
                # print(topic_to_remove_qoute_from)
                # print("topic_to_remove_qoute_from")

                for topic in topic_to_remove_qoute_from: 
                    # print(".................before removing Q ................................")
                    # print(topics_dict[topic]["quotes"])
                    topics_dict[topic]["quotes"].remove(quote)
                    # print(".................After removing Q .................................")
                    # print(topics_dict[topic]["quotes"])
            except Exception as e:
                print(f"Exception in the loop :  {e}")
                # printing stack trace 
                traceback.print_exc() 

        return topics_dict
    
    
    except Exception as e:
        print(f"An exception occured: {e}")
        # printing stack trace 
        traceback.print_exc() 
        return topics_dict

    



def make_transcript_flowful(ordered_topics, qa_draft):
    """
    Make the generated QA draft flowful in a continous interview conversation style

    Args:
    - ordered_topics: a list of topics as they should appear in the QA draft
    - qa_draft: a string of the selected and updated QA pairs


    Returns:
    - final_transcript: a string of the final QA pairs
    
    """

    messages = [
        {"role": "system",
         "content" : "You're a helpful editor's assistant."},
        {
            "role": "user",
            "content" : """

            <START OF Q/A pairs transcript> """ + 
            qa_draft
            +"""
            <END OF Q/A pairs transcript>

            <START OF ORDERED TOPICS>
            -
            """ 
            +
            "-".join(list(ordered_topics))
            +"""
            <END OF ORDERED TOPICS>

            Given the above Q/A pairs and the above list of ordred topics, your mission is to follow the bellow instructions:
            
            <INSTRUCTIONS START> 
            - Reorder the Q/A pairs following the exact order of the topics. 
            - make the question direct and precise.
            - Don't add "thank you" to the interviewer questions.
            - Avoid repeating the questions in the same style( avoid the "Thank you, now let's ...." style in the questions)
            - Do not say thank you in the questions
            - Do not use "Let's talk, Let's delve, ect" in the interviewer questions
            - Do not alter or modify the quotes.
            - Remove repetitive Q/A pairs (keep only the first original instance)
            - Do not mention the topics, the output must be a flowful interview transcript of question/answer pairs.
            - Each interviewer intercation should contain a full question.
            - Thank the interviewee at the end of the transcript for his time.
            - Ensure that  each QA pair in the input is also present in the output which is the flowful interview transcript to be generated.
            - Ensure that the QA pairs appear in the flawful interview transcript following the order of the topics in the input.
            - Ensure that the generated interview transcript is a flawfull, organized transcript that seems like a well planned interaction from beginning to end.
            <INSTRUCTIONS END> 

            """
        }
    ]


    completion = client.chat.completions.create(
    model="gpt-4o-mini", 
    messages = messages,
    temperature=0.5,
    max_tokens=5000,
    top_p=0.00001,
    frequency_penalty=0,
    presence_penalty=0,
    )

    return completion.choices[0].message.content



def prepare_quotes_options_str(redundant_quotes_dict):
    """
    This function takes the dictionnary of the redundant quotes and returns a string made out of it.

    Args:
    - redundant_quotes_dict: dictionnary of quotes as keys and topics as values

    Returns:
    - string : The original dictionnary of topics in a string format
    """
    final_string = ""
    for quote in redundant_quotes_dict.keys():
        topics = redundant_quotes_dict[quote]
        final_string += quotes_topic_str(quote, topics) 

    return final_string


def update_topic_assignment_all_at_once(redundant_quotes_dict, topics_dict, topics):
    """
    This function takes the dictionnary of the redundant quotes and returns corrected quotes to topics assignment

    Args:
    - redundant_quotes_dict: dictionnary of quotes as keys and topics as values
    - topics_dict: The original dictionnary of topics with redunduncy

    Returns:
    - updated_topic_assignment_dict = The original dictionnary of topics  without redunduncy
    """
    quotes_options_string = prepare_quotes_options_str(redundant_quotes_dict)


    messages = [

    {
        "role": "system",
        "content": "You are a helpfull assistant and JSON formater that helps editors assign quotes to the most appropriate topic option."
    },

    {
        "role": "user",
        "content":
        """


        < START of List Topic >
        """ +
        "\n".join(topics)
        + """
        <END of List topics >

        <START of quotes-topics options >
        """ +
        quotes_options_string
        + """
        <END of quotes-topics options >

        Given the above quotes and a set of topic options for each quote do the following:

        <INSTRUCTIONS START>

        1. For each quote, identify up to three topics that each quote relates to the most from the list of options.
        2. Return a dictionnary in such a way that each quote(key) has up to three values(three topics, which is the one the quote relates to the most)
        3. Return the quotes exactly as they were inputed, do not modify the quotes 
        4. The selected topic should be returned in the same way it was inputed (include topic number detail)
        5. Limit the topic selection to the list of topics attributed to each quote.
        6. Do not attribute to a quote a topic that doesnt exist in its related list of topics.
        7. Do not start the topic by - 
        8. Ensure that each unique topic has at least one quote attributed to it.
        9. Ensure that at least one quote is attributed to all topics that appeared in the list of topics defined by tags.
        10. Ensure the generated content is returned in a valid JSON format as illustarted bellow:

            <OUTPUT FORMAT START>

            { // Opening curly brace

            <quote text> : <topic>,
            <quote text> : <topic>,
            ...
            
            } // closing curly brace 
            <OUTPUT FORMAT END>

        - Ensure the generated content is returned in a valid JSON format
        <INSTRUCTIONS END>

        Follow the bellow example instance: 

        
        <START OF TOPIC SEELCTION EXAMPLE>
        <Quote> :
            Dr. Fong: This is a question I get essentially, four or five times a week, when people will say, do I have a gambling addiction? Or how do I know if my husband or wife or son or daughter is going to develop a gambling addiction? And I usually start with the following. First, I tell people gambling is part of something we do every day in our lives. It’s risk-taking. It’s decision-making. It’s going for rewards.

            Essentially, though, the difference is that between someone who gambles regularly and socially versus someone with a (quote) gambling disorder, or a gambling addiction is that if their gambling continues to bring harmful consequences to their lives and they continue to engage in gambling, that’s an addiction. Furthermore, men and women with gambling disorder, they experience all sorts of things that people who gamble recreationally do not. They have urges and cravings that get in the way of them 
            completing their daily lives. They have restrictions and limitations [00:05:00] on what they can do in life because of the consequences of gambling.

            So, much like any other addictive disorder, it isn’t so much how much you gamble or how often or how much you’ve lost, it’s what are the consequences and what are the 
            biological and psychological and social experiences it is for the person who is gambling? If they gamble in a way that’s harmful and distressing and emotionally painful, that’s an addiction. If they gamble and they have a lot of fun and they lose a lot of money, but it doesn’t impact their daily functioning, that’s not an addiction. That’s just a hobby.
            <Topic Options For above quote> :
            -Topic1: Definition of gambling addiction and its distinguishing factors from normal gambling.
            -Topic3: Classification of gambling addiction as an addictive disorder.


        LLM OUTPUT:

        {
            "Dr. Fong: This is a question I get essentially, four or five times a week, when people will say, do I have a gambling addiction? Or how do I know if my husband or wife or son or daughter is going to develop a gambling addiction? And I usually start with the following. First, I tell people gambling is part of something we do every day in our lives. It’s risk-taking. It’s decision-making. It’s going for rewards.

            Essentially, though, the difference is that between someone who gambles regularly and socially versus someone with a (quote) gambling disorder, or a gambling addiction is that if their gambling continues to bring harmful consequences to their lives and they continue to engage in gambling, that’s an addiction. Furthermore, men and women with gambling disorder, they experience all sorts of things that people who gamble recreationally do not. They have urges and cravings that get in the way of them 
            completing their daily lives. They have restrictions and limitations [00:05:00] on what they can do in life because of the consequences of gambling.

            So, much like any other addictive disorder, it isn’t so much how much you gamble or how often or how much you’ve lost, it’s what are the consequences and what are the 
            biological and psychological and social experiences it is for the person who is gambling? If they gamble in a way that’s harmful and distressing and emotionally painful, that’s an addiction. If they gamble and they have a lot of fun and they lose a lot of money, but it doesn’t impact their daily functioning, that’s not an addiction. That’s just a hobby."

            :
            "Topic1: Definition of gambling addiction and its distinguishing factors from normal gambling."
        }


        <END OF TOPIC SEELCTION EXAMPLE>
        """
    }
    ]


    completion = client.chat.completions.create(

        model = "gpt-4o-mini",
        temperature=0.4,
        max_tokens= 16000,
        frequency_penalty= 0,
        presence_penalty= 0,
        messages=messages,
        response_format={"type": "json_object"}
    )

    try:
        corrected_assignment = json.loads(completion.choices[0].message.content , strict=False)
    except Exception as e:
        print(e)
        corrected_assignment = {}
    

    for quote in redundant_quotes_dict.keys():
        try:
            pattern = re.escape(quote[:20])

            # Find the topic using re
            for q in corrected_assignment.keys():
                if re.match(pattern, q[:20]) != None:
                    quote_ = q
                    break

            topic_to_remove_qoute_from =  list(set(redundant_quotes_dict[quote]) - set([corrected_assignment[quote_]]))

            for topic in topic_to_remove_qoute_from: 
                topics_dict[topic]["quotes"].remove(quote)
        except Exception as e:
            print(f"Exception in the loop :  {e}")
            # printing stack trace 
            traceback.print_exc() 

    return topics_dict
