from .repeated_phrase import is_repetitive_japanese
from tqdm import tqdm


def clean_text_list(text_list):
    text_list = list(set(text_list))
    cleaned_text_list = []
    for text in tqdm(text_list):
        try:
            cleaned_text = is_repetitive_japanese(text)
        except Exception as e:
            print(e)
            clean_text_list=""
        if cleaned_text == "":
            continue
        cleaned_text_list.append(cleaned_text)
    return cleaned_text_list
