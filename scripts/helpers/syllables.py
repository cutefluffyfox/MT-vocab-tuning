import re


class SyllablesTokenizer:
    def __init__(self, lang: str):
        self.lang = lang

    def split_text_to_syllables(self, text: str) -> list[list[str]]:
        words = re.split('(\W)', text)
        output = []
        for word in words:
            output.append(self.split_word_to_syllables(word))
        return output

    def split_word_to_syllables(self, word: str) -> list[str]:
        raise NotImplemented('Split to syllables method is not implemented')


class TatarSyllablesTokenizer(SyllablesTokenizer):
    def __init__(self):
        super().__init__(lang='tt')
        self.alphabet = set('аәбвгдеёжҗзийклмнңоөпрстуүфхһцчшщъыьэюя')
        self.vowels = set('аәеёиоөуүыэюя')
        self.consonants = self.alphabet - self.vowels
        self.syllables_patterns = [
            'CVCC',
            'VCC',
            'CVC',
            'VC',
            'CV',
            'V'
        ]  # https://elibs.kai.ru/_docs_file/792690/HTML/14/


    def split_word_to_syllables(self, text: str):
        pass


def split_tatar_syllables_v1(word):
    # Define Tatar vowels (both Latin and Cyrillic alphabets)
    vowels = "аәеёэоөуүыиюяАӘЕЁЭОӨУҮЫИЮЯaəeoöuüıiAƏEOÖUÜIİ"

    syllables = []
    current_syllable = ""
    i = 0

    while i < len(word):
        current_syllable += word[i]

        # If we've just added a vowel and there's more characters
        if word[i] in vowels and i + 1 < len(word):
            # If the next character is a vowel, close this syllable
            if word[i + 1] in vowels:
                syllables.append(current_syllable)
                current_syllable = ""

            # If the next character is a consonant
            else:
                # Look ahead to see if there's another consonant followed by a vowel
                if i + 2 < len(word) and word[i + 2] in vowels and word[i + 1] not in vowels:
                    syllables.append(current_syllable)
                    current_syllable = ""
                # If we have two consonants in a row, close syllable after the first vowel
                elif i + 2 < len(word) and word[i + 1] not in vowels and word[i + 2] not in vowels:
                    syllables.append(current_syllable)
                    current_syllable = ""

        i += 1

    # Add the last syllable if there's anything left
    if current_syllable:
        syllables.append(current_syllable)

    return syllables



def split_tatar_syllables_v2(word):
    # Define vowels in Tatar
    vowels = 'аәеёэоөуүыиюя'
    result = []
    current_syllable = ""

    i = 0
    while i < len(word):
        current_syllable += word[i]

        # If we have a vowel in the current syllable and there are more characters
        if any(v in current_syllable.lower() for v in vowels) and i + 1 < len(word):
            next_char = word[i + 1]

            # If the next character is a vowel, close the current syllable
            if next_char.lower() in vowels:
                result.append(current_syllable)
                current_syllable = ""

            # If we have consonant followed by another consonant, split accordingly
            elif (i + 2 < len(word) and
                  next_char.lower() not in vowels and
                  word[i + 2].lower() not in vowels):
                result.append(current_syllable + next_char)
                current_syllable = ""
                i += 1

        i += 1

    # Add the last syllable if it's not empty
    if current_syllable:
        result.append(current_syllable)

    return result


def split_tatar_syllables_v3(word):
    # Define vowels in Tatar
    vowels = 'аәеёэоөуүыиюя'
    syllables = []
    i = 0

    # Find positions of all vowels
    vowel_positions = [i for i, char in enumerate(word) if char.lower() in vowels]

    if not vowel_positions:  # If no vowels, return the whole word
        return [word]

    # Process each vowel position to determine syllable boundaries
    for j in range(len(vowel_positions)):
        vowel_pos = vowel_positions[j]

        # For the last vowel
        if j == len(vowel_positions) - 1:
            # From current position to the end of the word
            syllables.append(word[i:])
            break

        # For other vowels, find the next boundary
        next_vowel_pos = vowel_positions[j + 1]

        # If there are consonants between vowels
        if next_vowel_pos - vowel_pos > 1:
            # The basic rule: consonant before a vowel belongs to that vowel's syllable
            boundary = next_vowel_pos
            # If there are multiple consonants, the boundary is after the first consonant
            if next_vowel_pos - vowel_pos > 2:
                boundary = vowel_pos + 2

            syllables.append(word[i:boundary])
            i = boundary
        else:
            # No consonants between vowels
            syllables.append(word[i:vowel_pos + 1])
            i = vowel_pos + 1

    return syllables


def split_tatar_syllables_v4(word):
    # Define vowels in Tatar
    vowels = 'аәеёэоөуүыиюя'
    syllables = []

    i = 0
    while i < len(word):
        # Find the next vowel from current position
        vowel_idx = -1
        for j in range(i, len(word)):
            if word[j].lower() in vowels:
                vowel_idx = j
                break

        # If no vowel found, take the rest as one syllable and exit
        if vowel_idx == -1:
            if i < len(word):
                syllables.append(word[i:])
            break

        # Start creating a syllable from this position
        current_syllable_start = i

        # Move past the vowel
        i = vowel_idx + 1

        # Check how many consonants follow the vowel
        consonant_count = 0
        while i < len(word) and word[i].lower() not in vowels:
            consonant_count += 1
            i += 1

        # Apply Tatar syllable pattern rules
        if consonant_count >= 2:
            # For CVCC or VCC patterns, include only one consonant in this syllable
            syllables.append(word[current_syllable_start:vowel_idx + 2])
            i = vowel_idx + 2  # Move past vowel + one consonant
        else:
            # For CV, V, CVC, or VC patterns, include all consonants until next vowel
            end_idx = i
            syllables.append(word[current_syllable_start:end_idx])

    return syllables


# Test examples
test_words = [
    "сәлам",  # hello
    "рәхмәт",  # thank you
    "китап",  # book
    "мәктәп",  # school
    "татарча",  # Tatar language
    "балалар",  # children
    "университет",  # university
    "тормыш"  # life
]

for word in test_words:
    syllables = split_tatar_syllables_v4(word)
    print(f"{word}: {'-'.join(syllables)}")