import json
from datasets import load_dataset
import random
random.seed(0)


codes = ["en", "bg", "cs", "de", "es", "fi", "fr", "pt", "ru", "zh"]

crosslingual_answer_template = {
    "en": {    
        "en": "Please answer in English.",
        "bg": "Please answer in Bulgarian.",
        "cs": "Please answer in Czech.",
        "de": "Please answer in German.",
        "es": "Please answer in Spanish.",
        "fi": "Please answer in Finnish.",
        "fr": "Please answer in French.",
        "pt": "Please answer in Portuguese.",
        "ru": "Please answer in Russian.",
        "zh": "Please answer in Chinese."
    },
    "zh": {
        "en": "请用英语回答。",
        "bg": "请用保加利亚语回答。",
        "cs": "请用捷克语回答。",
        "de": "请用德语回答。",
        "es": "请用西班牙语回答。",
        "fi": "请用芬兰语回答。",
        "fr": "请用法语回答。",
        "pt": "请用葡萄牙语回答。",
        "ru": "请用俄语回答。",
        "zh": "请用中文回答。"
    },
    "es": {
        "en": "Por favor, responde en inglés.",
        "bg": "Por favor, responde en búlgaro.",
        "cs": "Por favor, responde en checo.",
        "de": "Por favor, responde en alemán.",
        "es": "Por favor, responde en español.",
        "fi": "Por favor, responde en finlandés.",
        "fr": "Por favor, responde en francés.",
        "pt": "Por favor, responde en portugués.",
        "ru": "Por favor, responde en ruso.",
        "zh": "Por favor, responde en chino."
    },
    "bg": {
        "en": "Моля, отговорете на английски.",
        "bg": "Моля, отговорете на български.",
        "cs": "Моля, отговорете на чешки.",
        "de": "Моля, отговорете на немски.",
        "es": "Моля, отговорете на испански.",
        "fi": "Моля, отговорете на финландски.",
        "fr": "Моля, отговорете на френски.",
        "pt": "Моля, отговорете на португалски.",
        "ru": "Моля, отговорете на руски.",
        "zh": "Моля, отговорете на китайски."
    },
    "cs": {
        "en": "Prosím, odpovězte anglicky.",
        "bg": "Prosím, odpovězte bulharsky.",
        "cs": "Prosím, odpovězte česky.",
        "de": "Prosím, odpovězte německy.",
        "es": "Prosím, odpovězte španělsky.",
        "fi": "Prosím, odpovězte finsky.",
        "fr": "Prosím, odpovězte francouzsky.",
        "pt": "Prosím, odpovězte portugalsky.",
        "ru": "Prosím, odpovězte rusky.",
        "zh": "Prosím, odpovězte čínsky."
    },
    "de": {
        "en": "Bitte antworten Sie auf Englisch.",
        "bg": "Bitte antworten Sie auf Bulgarisch.",
        "cs": "Bitte antworten Sie auf Tschechisch.",
        "de": "Bitte antworten Sie auf Deutsch.",
        "es": "Bitte antworten Sie auf Spanisch.",
        "fi": "Bitte antworten Sie auf Finnisch.",
        "fr": "Bitte antworten Sie auf Französisch.",
        "pt": "Bitte antworten Sie auf Portugiesisch.",
        "ru": "Bitte antworten Sie auf Russisch.",
        "zh": "Bitte antworten Sie auf Chinesisch."
    },
    "fi": {
        "en": "Vastaa englanniksi, kiitos.",
        "bg": "Vastaa bulgariaksi, kiitos.",
        "cs": "Vastaa tšekiksi, kiitos.",
        "de": "Vastaa saksaksi, kiitos.",
        "es": "Vastaa espanjaksi, kiitos.",
        "fi": "Vastaa suomeksi, kiitos.",
        "fr": "Vastaa ranskaksi, kiitos.",
        "pt": "Vastaa portugaliksi, kiitos.",
        "ru": "Vastaa venäjäksi, kiitos.",
        "zh": "Vastaa kiinaksi, kiitos."
    },
    "fr": {
        "en": "Veuillez répondre en anglais.",
        "bg": "Veuillez répondre en bulgare.",
        "cs": "Veuillez répondre en tchèque.",
        "de": "Veuillez répondre en allemand.",
        "es": "Veuillez répondre en espagnol.",
        "fi": "Veuillez répondre en finnois.",
        "fr": "Veuillez répondre en français.",
        "pt": "Veuillez répondre en portugais.",
        "ru": "Veuillez répondre en russe.",
        "zh": "Veuillez répondre en chinois."
    },
    "pt": {
        "en": "Por favor, responda em inglês.",
        "bg": "Por favor, responda em búlgaro.",
        "cs": "Por favor, responda em tcheco.",
        "de": "Por favor, responda em alemão.",
        "es": "Por favor, responda em espanhol.",
        "fi": "Por favor, responda em finlandês.",
        "fr": "Por favor, responda em francês.",
        "pt": "Por favor, responda em português.",
        "ru": "Por favor, responda em russo.",
        "zh": "Por favor, responda em chinês."
    },
    "ru": {
        "en": "Пожалуйста, ответьте на английском.",
        "bg": "Пожалуйста, ответьте на болгарском.",
        "cs": "Пожалуйста, ответьте на чешском.",
        "de": "Пожалуйста, ответьте на немецком.",
        "es": "Пожалуйста, ответьте на испанском.",
        "fi": "Пожалуйста, ответьте на финском.",
        "fr": "Пожалуйста, ответьте на французском.",
        "pt": "Пожалуйста, ответьте на португальском.",
        "ru": "Пожалуйста, ответьте на русском.",
        "zh": "Пожалуйста, ответьте на китайском."
    }
}

translate_template = {
    "en": "Please translate the query to English.",
    "bg": "Моля, преведете въпроса на английски.",
    "cs": "Přeložte prosím otázku do angličtiny.",
    "de": "Bitte übersetzen Sie die Frage ins Englische.",
    "es": "Por favor, traduzca la pregunta al inglés.",
    "fi": "Käännä kysymys englanniksi.",
    "fr": "Veuillez traduire la requête en anglais.",
    "pt": "Por favor, traduza a consulta para inglês",
    "ru": "Пожалуйста, переведите запрос на английский.",
    "zh": "请把问题翻译成英文。",
}

translate_en_then_answer_template = {
    "en": "Please translate the query to English and respond in English.",
    "bg": "Моля, преведете въпроса на английски и отговорете на английски.",
    "cs": "Přeložte prosím otázku do angličtiny a odpovězte v angličtině.",
    "de": "Bitte übersetzen Sie die Frage ins Englische und beantworten Sie sie auf Englisch.",
    "es": "Por favor, traduzca la pregunta al inglés y responda en inglés.",
    "fi": "Käännä kysymys englanniksi ja vastaa englanniksi.",
    "fr": "Veuillez traduire la requête en anglais et répondre en anglais.",
    "pt": "Traduza o pedido para inglês e responda em inglês.",
    "ru": "Пожалуйста, переведите запрос на английский и ответьте на английском.",
    "zh": "请把问题翻译成英文并用英文回答。",
}


all_data = {}
for code in codes:
    with open(f"./alpaca_data_cleaned.{code}.json", "r") as json_file:
        data = json.load(json_file)
        # print(code, len(data))
    all_data[code] = data

filtered_data = {code:[] for code in codes}
for i, instance in enumerate(all_data["en"]):
    if "input" not in instance or instance["input"].strip() == "":
        for code in codes:
            filtered_data[code].append(all_data[code][i])

for code in all_data:
    print(code, filtered_data[code][-111])
    assert len(filtered_data[code]) == len(filtered_data[codes[0]])



alpaca_translate_en_then_answer_multi_turn_data = []
alpaca_translate_en_then_answer_single_turn_data = []
non_en_codes = codes[1:]
for i in range(len(filtered_data["en"])):
    instruction_code = random.choice(non_en_codes)
    if random.randint(0,1) == 0:
        translation_instruction = translate_template["en"]
        translate_then_answer_instruction = translate_en_then_answer_template["en"]
    else:
        translation_instruction = translate_template[instruction_code]
        translate_then_answer_instruction = translate_en_then_answer_template[instruction_code]
    instruction = filtered_data[instruction_code][i]["instruction"].strip()

    if instruction[-1] != "." and instruction[-1] != "?":
        delimiter = "\n"
    elif random.randint(0,1) == 0:
        delimiter = " "
    else:
        delimiter = "\n"

    new_instruction_translate = instruction + delimiter + translation_instruction
    new_instruction_translate_answer = instruction + delimiter + translate_then_answer_instruction
    
    alpaca_translate_en_then_answer_multi_turn_data.append(
        {
            "conversations": [
                {"from": "gpt-mt", "value": new_instruction_translate},
                {"from": "gpt", "value": filtered_data["en"][i]["instruction"].strip()},
                {"from": "human", "value": "Please answer the question."},
                {"from": "gpt", "value": filtered_data["en"][i]["output"].strip()}
            ],
            "dataset": "alpaca-translate-then-answer-multi-turn",
            "lang": instruction_code + "-" + "en",
            "task": "chat",
            "split": "train",
        }
    )
    
    alpaca_translate_en_then_answer_single_turn_data.append(
        {
            "conversations": [
                {"from": "gpt-mt", "value": new_instruction_translate_answer},
                {"from": "gpt", "value": "English Translation: " + filtered_data["en"][i]["instruction"].strip() + "\nAnswer: " + filtered_data["en"][i]["output"].strip()},
            ],
            "dataset": "alpaca-translate-then-answer-single-turn",
            "lang": instruction_code + "-" + "en",
            "task": "chat",
            "split": "train",
        }
    )

with open("alpaca_translate_en_then_answer_multi_turn.json", "w") as json_file:
    json.dump(alpaca_translate_en_then_answer_multi_turn_data, json_file, indent=4, ensure_ascii=False)

with open("alpaca_translate_en_then_answer_single_turn.json", "w") as json_file:
    json.dump(alpaca_translate_en_then_answer_single_turn_data, json_file, indent=4, ensure_ascii=False)

crosslingual_answer_data = []
for i in range(len(filtered_data["en"])):
    instruction_code = random.choice(codes)
    instruction = filtered_data[instruction_code][i]["instruction"].strip()
    
    cross_lingual_instruction_code = random.choice(codes)
    output_code = random.choice(codes)
    cross_lingual_instruction = crosslingual_answer_template[cross_lingual_instruction_code][output_code]
    
    if instruction[-1] != "." and instruction[-1] != "?":
        delimiter = "\n"
    elif random.randint(0,1) == 0:
        delimiter = " "
    else:
        delimiter = "\n"

    new_instruction = instruction + delimiter + cross_lingual_instruction
    output = filtered_data[output_code][i]["output"].strip()
    crosslingual_answer_data.append(
        {
            "conversations": [
                {"from": "gpt-mt-human", "value": new_instruction},
                {"from": "gpt", "value": output}
            ],
            "dataset": "alpaca-crosslingual-answer",
            "lang": instruction_code + "_" + cross_lingual_instruction_code + "-" + output_code,
            "task": "chat",
            "split": "train",
        }
    )

with open("alpaca_crosslingual_answer.json", "w") as json_file:
    json.dump(crosslingual_answer_data, json_file, indent=4, ensure_ascii=False)
    

assert len(crosslingual_answer_data) == len(alpaca_translate_en_then_answer_multi_turn_data) == len(alpaca_translate_en_then_answer_single_turn_data)