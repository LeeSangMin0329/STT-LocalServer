from faster_whisper.transcribe import Segment
from typing import Iterable


remove_words = ["ご視聴ありがとうござ"]

# 중복 및 특정 문자열 포함 아이템 삭제 후 문자열 병합
def clean_iterable(iterable: Iterable[Segment]):
    result = []
    seen_word = set()

    for word in iterable:
        text = word.text

        if text in seen_word:
            continue

        is_bad_sentence = False

        for bad_word in remove_words:
            if bad_word in text:
                is_bad_sentence = True
                break
        
        if is_bad_sentence:
            continue

        result.append(text)
        seen_word.add(text)

    return ' '.join(result)
