"""
Streamlit app with a top-right settings icon that opens a drawer of editable prompts.
Run with: streamlit run app.py
"""

import json
import os
from datetime import datetime
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit_local_storage import LocalStorage

load_dotenv()

localS = LocalStorage()  # this is a Streamlit component

DEFAULT_PROMPTS = [
    {
        "label": "Generate brand data",
        "text": (
            "Find out the following brand's URL (homepage) and description.\n"
            "This is a real brand, so you should go online and look it up before answering.\n"
            "If you can't find the exact URL, return the closest one.\n\n"
            "Brand name: {brand}\n"
            "{hint}\n\n"
            "Return a JSON object:```json\n"
            "{\n"
            "    \"brand_url\": \"...\",\n"
            "    \"brand_description\": \"...\"\n"
            "}\n"
            "```\n\n"
            "Write both fields in {language}, as perceived from {city}{country}."
        ),
    },
    {
        "label": "Infer industry",
        "text": (
            "Identify the most specific industry category this brand operates in.\n\n"
            "Brand name: {brand}\n"
            "Brand description:\n"
            "{description}\n\n"
            "Return a JSON object:\n"
            "```json\n"
            "{\n"
            "    \"industry_name\": \"...\",\n"
            "    \"industry_description\": \"...\"\n"
            "}\n"
            "```\n\n"
            "Focus on the *primary industry*, not the brand itself.\n"
            "The response *must not include any brand, company, or product names*,  only general industry terms and descriptions.\n"
            "When multiple categories apply, choose the most specific one that directly reflects the brand’s core activity.\n"
            "Write both fields in {language}, as perceived from {city}{country}."
        ),
    },
    {
        "label": "Generate topics", 
        "text": (
            "List 3 short, relevant topics people might discuss with GPTs about this industry, as they look for product recomandations in this space.\n\n"
            "Industry: {industry}\n"
            "Industry description:\n"
            "{industry_description}\n\n"
            "Return a JSON object:```json\n"
            "{\n"
            "    \"topics\": [\"...\", \"...\", \"...\"]\n"
            "}\n"
            "```\n\n"
            "Each topic should be *concise (a few words)*, realistic, and relevant to the industry and country context.\n"
            "Do *not* include brand or company names.\n"
            "Write the topic names in {language}, as perceived from {city}{country}."
        ),
    },
    {
        "label": "Generate scenarios",
        "text": (
            "Generate 4 realistic questions a person might ask a GPT related to this industry and topic.\n"
            "Their goal is to get a product recommendation, not to explore the space in general.\n"
            "If you think it's needed, include the fact that they are from {city}{country} in the question.\n\n"
            "Industry: {industry}\n"
            "Topic: {topic}\n"
            "Industry description:\n"
            "{industry_description}\n\n"
            "Return a JSON object:```json\n"
            "{\n"
            "    \"questions\": [\"...\", \"...\", \"...\", \"...\"]\n"
            "}\n"
            "```\n\n"
            "Each question should:\n"
            "•	Be natural and relevant to the topic and industry.\n"
            "•	Reflect what people from {city}{country} might genuinely ask a GPT.\n"
            "•	Be written in {language}.\n"
            "•	Avoid any brand or company names.\n"
            "•	Be short and conversational, as if typed into ChatGPT or another assistant."
        )
    }
]


def get_stored_prompts():
    st.empty()
    stored = localS.getItem("prompts")
    if not stored:
        return None
    try:
        parsed = json.loads(stored)
    except Exception:
        return None
    if not isinstance(parsed, list):
        return None

    prompts = []
    for idx, prompt in enumerate(parsed):
        if not isinstance(prompt, dict):
            continue
        text = prompt.get("text")
        if not isinstance(text, str):
            continue
        label = prompt.get("label") or f"Prompt {idx + 1}"
        prompts.append({"label": label, "text": text})
    return prompts or None

def init_state():
    lastLoad = localS.getItem("last_load")
    now = datetime.now().timestamp()
    if lastLoad is not None:
        diff = now - lastLoad
        if diff > 60*60: # 1 hour
            localS.deleteAll()
    else:
        localS.deleteAll()

    localS.setItem("last_load", now, "set_last_load")

    stored_prompts = get_stored_prompts()
    base_prompts = stored_prompts or DEFAULT_PROMPTS

    if "prompts" not in st.session_state:
        st.session_state.prompts = [prompt.copy() for prompt in base_prompts]

    for idx, prompt in enumerate(st.session_state.prompts):
        key = f"prompt_{idx}"
        if key not in st.session_state:
            st.session_state[key] = prompt["text"]
    
    stored_brand_url = localS.getItem("brand_url")
    if stored_brand_url is None:
        stored_brand_url = "To Be Generated"
        localS.setItem("brand_url", stored_brand_url, key="set_brand_url_default")
    if "brand_url" not in st.session_state:
        st.session_state.brand_url = stored_brand_url
    
    stored_brand_description = localS.getItem("brand_description")
    if stored_brand_description is None:
        stored_brand_description = "To Be Generated"
        localS.setItem("brand_description", stored_brand_description, key="set_brand_description_default")
    if "brand_description" not in st.session_state:
        st.session_state.brand_description = stored_brand_description

    stored_industry = localS.getItem("industry")
    if stored_industry is None:
        stored_industry = "To Be Generated"
        localS.setItem("industry", stored_industry, key="set_industry_default")
    if "industry" not in st.session_state:
        st.session_state.industry = stored_industry

    stored_industry_description = localS.getItem("industry_description")
    if stored_industry_description is None:
        stored_industry_description = "To Be Generated"
        localS.setItem(
            "industry_description",
            stored_industry_description,
            key="set_industry_description_default",
        )
    if "industry_description" not in st.session_state:
        st.session_state.industry_description = stored_industry_description
    
    stored_topics = localS.getItem("topics")
    if stored_topics is None:
        stored_topics = "To Be Generated"
        localS.setItem("topics", stored_topics, key="set_topics_default")
    if "topics" not in st.session_state:
        st.session_state.topics = stored_topics

    stored_questions = localS.getItem("questions")
    if stored_questions is None:
        stored_questions = "To Be Generated"
        localS.setItem("questions", stored_questions, key="set_questions_default")
    if "questions" not in st.session_state:
        st.session_state.questions = stored_questions


def get_prompt_by_label(label):
    """Find a prompt by its label/key."""
    for prompt in st.session_state.prompts:
        if prompt.get("label") == label:
            return prompt.get("text")
    return None


def fill_prompt(template, **fields):
    """Replace only our known placeholders without touching other braces."""
    result = template
    for key, value in fields.items():
        result = result.replace(f"{{{key}}}", value)
    return result


def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Missing OPENAI_API_KEY. Add it to your .env file or environment.")
        return None
    return OpenAI(api_key=api_key)


def extract_json_object(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None
    return None


def response_output_to_text(response):
    """Extract text content from the Responses API output."""
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    output_items = getattr(response, "output", None)
    if not output_items:
        return None

    chunks = []
    for output_item in output_items:
        content_items = getattr(output_item, "content", None)
        if content_items is None and isinstance(output_item, dict):
            content_items = output_item.get("content")
        if not content_items:
            continue

        for content_item in content_items:
            item_type = getattr(content_item, "type", None)
            if item_type is None and isinstance(content_item, dict):
                item_type = content_item.get("type")
            if item_type not in ("output_text", "text"):
                continue

            item_text = getattr(content_item, "text", None)
            if item_text is None and isinstance(content_item, dict):
                item_text = content_item.get("text")
            if item_text:
                chunks.append(item_text)

    return "".join(chunks) if chunks else None


def persist_prompts():
    """Save current prompts to local storage."""
    try:
        localS.setItem("prompts", json.dumps(st.session_state.prompts), key="set_prompts")
    except Exception as exc:
        st.warning(f"Could not save prompts locally: {exc}")


def handle_prompt_change(idx):
    """Update a single prompt in state and persist."""
    key = f"prompt_{idx}"
    if idx < 0 or idx >= len(st.session_state.prompts):
        return
    st.session_state.prompts[idx]["text"] = st.session_state.get(key, "")
    persist_prompts()

def generate_brand_data(brand, hint, city, country, language):
    if len(city) > 0:
        city += ", "

    client = get_openai_client()
    if client is None:
        return None

    base_prompt = get_prompt_by_label("Generate brand data")
    if not base_prompt:
        st.error("Prompt 'Generate brand data' not found.")
        return None
    
    formatted_prompt = fill_prompt(
        base_prompt,
        brand=brand,
        hint=hint,
        city=city,
        country=country,
        language=language,
    )

    # First try to use the web tool (Responses API). Fall back to a plain chat completion on failure.
    content = None
    if getattr(client, "responses", None) is not None:
        try:
            web_response = client.responses.create(
                model="gpt-4.1",
                input=formatted_prompt,
                tools=[{"type": "web_search"}],
            )
            content = response_output_to_text(web_response)
            if not content:
                st.warning("Web tool response did not contain text. Falling back to base model.")
        except Exception as exc:
            st.warning(f"Web tool request failed. Falling back to base model. Details: {exc}")

    if content is None:
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.4,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            st.error(f"OpenAI request failed: {exc}")
            return None

        if not response.choices:
            st.error("OpenAI response was empty.")
            return None

        content = response.choices[0].message.content

    parsed = extract_json_object(content)
    if not parsed:
        st.error("Could not parse brand data from the OpenAI response.")
        return None

    brand_url = parsed.get("brand_url")
    brand_description = parsed.get("brand_description")
    if not brand_url or not brand_description:
        st.error("OpenAI response did not include brand_url and brand_description.")
        return None

    return {"brand_url": brand_url, "brand_description": brand_description}

def generate_industry_data(brand, description, city, country, language):
    if len(city) > 0:
        city += ", "

    client = get_openai_client()
    if client is None:
        return None

    base_prompt = get_prompt_by_label("Infer industry")
    if not base_prompt:
        st.error("Prompt 'Infer industry' not found.")
        return None
    
    formatted_prompt = fill_prompt(
        base_prompt,
        brand=brand,
        description=description,
        city=city,
        country=country,
        language=language,
    )

    content = None
    if getattr(client, "responses", None) is not None:
        try:
            web_response = client.responses.create(
                model="gpt-4.1",
                input=formatted_prompt,
                tools=[{"type": "web_search"}],
            )
            content = response_output_to_text(web_response)
            if not content:
                st.warning("Web tool response did not contain text. Falling back to base model.")
        except Exception as exc:
            st.warning(f"Web tool request failed. Falling back to base model. Details: {exc}")

    if content is None:
        try:
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[{"role": "user", "content": formatted_prompt}],
                temperature=0.4,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            st.error(f"OpenAI request failed: {exc}")
            return None

        if not response.choices:
            st.error("OpenAI response was empty.")
            return None

        content = response.choices[0].message.content
    parsed = extract_json_object(content)
    if not parsed:
        st.error("Could not parse industry data from the OpenAI response.")
        return None

    industry_name = parsed.get("industry_name")
    industry_description = parsed.get("industry_description")
    if not industry_name or not industry_description:
        st.error("OpenAI response did not include industry_name and industry_description.")
        return None

    return {"industry_name": industry_name, "industry_description": industry_description}

def generate_topics(industry, industry_description, city, country, language):
    if len(city) > 0:
        city += ", "

    client = get_openai_client()
    if client is None:
        return None

    base_prompt = get_prompt_by_label("Generate topics")
    if not base_prompt:
        st.error("Prompt 'Generate topics' not found.")
        return None
    
    formatted_prompt = fill_prompt(
        base_prompt, 
        industry=industry, 
        industry_description=industry_description, 
        city=city,
        country=country, 
        language=language
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        st.error(f"OpenAI request failed: {exc}")
        return None

    if not response.choices:
        st.error("OpenAI response was empty.")
        return None

    content = response.choices[0].message.content
    parsed = extract_json_object(content)
    if not parsed:
        st.error("Could not parse topics from the OpenAI response.")
        return None

    topics = parsed.get("topics")
    if not topics:
        st.error("OpenAI response did not include topics.")
        return None

    return {"topics": topics}

def generate_scenarios(industry, industry_description, topic, city, country, language):
    if len(city) > 0:
        city += ", "

    client = get_openai_client()
    if client is None:
        return None

    base_prompt = get_prompt_by_label("Generate scenarios")
    if not base_prompt:
        st.error("Prompt 'Generate scenarios' not found.")
        return None
    
    formatted_prompt = fill_prompt(
        base_prompt, 
        industry=industry, 
        industry_description=industry_description, 
        topic=topic, 
        city=city,
        country=country, 
        language=language
    )
    
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=[{"role": "user", "content": formatted_prompt}],
            temperature=0.4,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        st.error(f"OpenAI request failed: {exc}")
        return None

    if not response.choices:
        st.error("OpenAI response was empty.")
        return None

    content = response.choices[0].message.content
    parsed = extract_json_object(content)
    if not parsed:
        st.error("Could not parse scenarios from the OpenAI response.")
        return None

    questions = parsed.get("questions")
    if not questions:
        st.error("OpenAI response did not include questions.")
        return None

    return {"questions": questions}

def render_settings_drawer():
    # Widen popover to act like a drawer (~80% viewport width).
    st.markdown(
        """
        <style>
            [data-testid="stPopoverBody"] {
                width: 80vw !important;
                max-width: 80vw !important;
            }
            [data-testid="stPopover"] {
                inset: 10px 10px auto auto !important;
            }
            [data-testid="stPopoverContent"] {
                max-height: 85vh;
                overflow: auto;
            }
            /* Hide the streamlit-local-storage iframe completely */
            iframe[title="streamlit_local_storage.st_local_storage"] {
                display: none !important;
                height: 0 !important;
                width: 0 !important;
            }

            /* Also collapse its container so no extra space is shown */
            div[data-testid="stCustomComponentV1"] {
                margin: 0 !important;
                padding: 0 !important;
                height: 0 !important;
            }

        </style>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns([0.8, 0.2])
    with cols[1]:
        with st.popover("⚙️", use_container_width=True):
            st.write("Prompts")
            current_prompts = []
            for idx, prompt in enumerate(st.session_state.prompts):
                key = f"prompt_{idx}"
                label = prompt.get("label") or f"Prompt {idx + 1}"
                updated_text = st.text_area(
                    label,
                    key=key,
                    height=160,
                    on_change=handle_prompt_change,
                    args=(idx,),
                )
                current_prompts.append({"label": label, "text": updated_text})
            st.session_state.prompts = current_prompts

def main():
    st.set_page_config(
        page_title="Prompt Drawer",
        layout="wide",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": None,
        },
    )
    init_state()

    render_settings_drawer()
    st.title("Main")

    stored_brand_name = localS.getItem("brand_name")
    if stored_brand_name is None:
        stored_brand_name = "NatWest"
        localS.setItem("brand_name", stored_brand_name, key="set_brand_name_default")
    brand_name = st.text_input("Brand name", key="brand_name", value=stored_brand_name, placeholder="Enter brand name")
    if brand_name != stored_brand_name:
        localS.setItem("brand_name", brand_name, key="set_brand_name")

    stored_brand_hint = localS.getItem("brand_hint")
    if stored_brand_hint is None:
        stored_brand_hint = ""
        localS.setItem("brand_hint", stored_brand_hint, key="set_brand_hint_default")
    brand_hint = st.text_input("Brand hint", key="brand_hint", value=stored_brand_hint, placeholder="Optional. Write down anything to help us getter identify your brand.")
    if brand_hint != stored_brand_hint:
        localS.setItem("brand_hint", brand_hint, key="set_brand_hint")

    stored_country = localS.getItem("country")
    countries = [
        "United States",
        "United Kingdom",
        "Romania",
        "Canada",
        "Croatia",
        "Kazakhstan",
        "Australia",
        "Germany",
        "France",
        "Spain",
        "Italy",
        "Netherlands",
        "Sweden",
        "Brazil",
        "India",
        "Japan",
        "Singapore",
    ]
    if stored_country not in countries:
        stored_country = "United Kingdom"
        localS.setItem("country", stored_country, key="set_country_default")

    stored_city = localS.getItem("city")
    if stored_city is None:
        stored_city = ""
        localS.setItem("city", stored_city, key="set_city_default")

    stored_language = localS.getItem("language")
    languages = [
        "English",
        "Spanish",
        "French",
        "German",
        "Italian",
        "Dutch",
        "Swedish",
        "Portuguese",
        "Romanian",
        "Croatian",
        "Kazakh",
        "Russian",
        "Hindi",
        "Japanese",
        "Chinese",
        "Korean",
    ]
    if stored_language not in languages:
        stored_language = "English"
        localS.setItem("language", stored_language, key="set_language_default")

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_country = st.selectbox(
            "Country",
            countries,
            index=countries.index(stored_country),
            key="country",
        )
        if selected_country != stored_country:
            localS.setItem("country", selected_country, key="set_country")
    with col2:
        city = st.text_input("City", key="city", value=stored_city, placeholder="Optional: City name")
        if city != stored_city:
            localS.setItem("city", city, key="set_city")
    with col3:
        selected_language = st.selectbox(
            "Language",
            languages,
            index=languages.index(stored_language),
            key="language",
        )
        if selected_language != stored_language:
            localS.setItem("language", selected_language, key="set_language")

    st.divider()
    
    if st.button("Generate Brand Data", type="primary"):
        with st.spinner("Generating brand data..."):
            result = generate_brand_data(
                brand=brand_name,
                hint=brand_hint,
                city=city,
                country=selected_country,
                language=selected_language,
            )
        if result:
            st.session_state.brand_url = result["brand_url"]
            st.session_state.brand_description = result["brand_description"]
            localS.setItem("brand_url", result["brand_url"], key="set_brand_url")
            localS.setItem("brand_description", result["brand_description"], key="set_brand_description")

    st.text_input("Brand URL", key="brand_url")
    st.text_area("Brand description", key="brand_description", height=120)

    st.divider()
    if st.button("Generate Industry Data", type="primary"):
        with st.spinner("Generating industry data..."):
            result = generate_industry_data(
                brand=brand_name,
                description=st.session_state.brand_description,
                city=city,
                country=selected_country,
                language=selected_language,
            )
        if result:
            st.session_state.industry = result["industry_name"]
            st.session_state.industry_description = result["industry_description"]
            localS.setItem("industry", result["industry_name"], key="set_industry")
            localS.setItem(
                "industry_description",
                result["industry_description"],
                key="set_industry_description",
            )

    st.text_input("Industry", key="industry", disabled=True)
    st.text_area(
        "Industry description",
        key="industry_description",
        disabled=True,
        height=120,
    )

    st.divider()
    if st.button("Generate Topics", type="primary"):
        with st.spinner("Generating topics..."):
            result = generate_topics(
                industry=st.session_state.industry,
                industry_description=st.session_state.industry_description,
                city=city,
                country=selected_country,
                language=selected_language,
            )
        if result:
            topics_list = result["topics"]
            topics_str = "\n".join(topics_list) if isinstance(topics_list, list) else str(topics_list)
            st.session_state.topics = topics_str
            localS.setItem("topics", topics_str, key="set_topics")

    st.text_area(
        "Topics", 
        key="topics", 
        height=120,
    )

    st.divider()
    if st.button("Generate Scenarios (for the 1st topic)", type="primary"):
        with st.spinner("Generating scenarios..."):
            result = generate_scenarios(
                industry=st.session_state.industry,
                industry_description=st.session_state.industry_description,
                topic=st.session_state.topics.split("\n")[0],
                city=city,
                country=selected_country,
                language=selected_language,
            )
        if result:
            questions_list = result["questions"]
            questions_str = "\n\n".join(f"- {question}" for question in questions_list) if isinstance(questions_list, list) else str(questions_list)
            st.session_state.questions = questions_str
            localS.setItem("questions", questions_str, key="set_questions")
    
    st.text_area(
        "Scenarios",
        key="questions",
        height=240,
    )

# Streamlit runs the script on every interaction, so we call main() directly
# instead of using if __name__ == "__main__" which can cause double execution
main()
