#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""garbage code to make garbage text"""

# import nltk
# nltk.download('wordnet')
# nltk.download('cmudict')
# TODO: another download was needed for nltk for this dunno what thou
import sqlite3
from typing import List, Optional, Set
import re

import csv

import random
import string
from math import ceil

import nltk
import pandas
import pronouncing
from textblob import TextBlob, Word, Sentence


from nltk.corpus import wordnet as wn
from nltk.tokenize.treebank import TreebankWordDetokenizer
from better_profanity import profanity
import lorem
import splitter


def knotter(token: str) -> str:
    token = re.sub(
        r"no+t",
        lambda match: f"kn{'o' * random.choice(range(1, 3))}t",
        token,
        flags=re.IGNORECASE,
    )
    return token


# TODO: refactor this global garbage

common_mispellings_probability = 0.2
hard_owo_replace_probability = 0.2

bold_text_probability = 0.04

REEE_probability = 0.06
REEE_allcaps_probability = 0.3

add_random_rp_action = True
add_random_rp_mid_sentance_action_probability = 0.04
add_random_rp_end_sentance_action_probability = 0.08
more_verbs_probability_decay = 0.4

add_random_garbage = True
add_random_garbage_probability = 0.01

add_random_plurals = True
add_random_plurals_probability = 0.1

randomly_lemmatize = True
randomly_lemmatize_probability = 0.1

randomly_overemphasis_punct = True
randomly_overemphasis_punct_probability = 0.5
randomly_overemphasis_punct_max_fuck = 4

randomly_capitalize_word = True
randomly_capitalize_word_probability = 0.1

randomly_spongebob_word = True
randomly_spongebob_word_probability = 0.1

add_randomly_text_face_emoji = True
add_randomly_text_face_emojis_probability = 0.05

add_random_simple_text_emoji = True
add_random_simple_text_emoji_probability = 0.07

randomly_swap_char = True
randomly_swap_char_probability = 0.04
randomly_swap_char_swap_percent = 0.2

randomly_insert_char = True
randomly_insert_char_probability = 0.04
randomly_insert_char_insert_percent = 0.1

random_leet_speek = True
random_leet_speek_probability = 0.1

utf_8_char_swaps_probability = 0.1

random_censor_probability = 0.01
random_censor_percent = 0.25

censor_profanity_probability = 0.7
censor_profanity_percent = 0.25

random_synonym_probability = 0.5

random_ending_y_probabilty = 0.05
leet_min_token_lenght = 5

add_definition_in_parenthesis_probability = 0.005
adding_ending_ksksk_andioop_probability = 0.8
adding_ending_ksksk_save_the_turtles_probability = 0.3

ksks_enlargement_probability = 0.7

owo_vs_ouo_bais = 0.5

random_lorem_ipsum_probabilty = 0.015
lorem_ipsum_fuck_probabilty = 0.3

add_extra_ed_probability = 0.05
split_compound_word_probability = 0.03

lazy_char_subbing_probability = 0.6
uck_to_ucc_swap_probability = 0.4

juwuice_swap_probability = 0.5

add_x3_if_token_has_rawr_probability = 0.2
me_2_meh_swap_probability = 0.5
me_2_meow_swap_probability = 0.5

hard_uwu_replace_probability = 0.3
sub_to_subby_swap_probability = 0.3


def owoer(token: str) -> str:
    token = re.sub(
        r"(ou)([^o])?",
        lambda match: f"ouo{match.group(2) or ''}",
        token,
        flags=re.IGNORECASE,
    )
    token = re.sub(
        r"(ow)([^o])?",
        lambda match: f"owo{match.group(2) or ''}",
        token,
        flags=re.IGNORECASE,
    )
    token = re.sub(
        r"(ov)([^o])?",
        lambda match: f"ovo{match.group(2) or ''}",
        token,
        flags=re.IGNORECASE,
    )

    token = re.sub(r"(cor)", lambda match: f"cowor", token)

    if (
        "owo" not in token.lower()
        and "ouo" not in token.lower()
        and decision(hard_owo_replace_probability)
    ):
        owo_str = "owo" if decision(owo_vs_ouo_bais) else "ouo"
        token = re.sub(
            r"(o+)", lambda match: (owo_str * len(match.group(1))).replace("oo", "o"), token, flags=re.IGNORECASE, count=random.choice(range(0, 2))
        )

    # TODO: UWU
    # juice -> juwuice
    if decision(juwuice_swap_probability):
        token = re.sub(
            r"u+(i?ce)",
            lambda match: f"uwu{match.group(1)}",
            token,
            flags=re.IGNORECASE,
        )

    if "uwu" not in token.lower() and decision(hard_uwu_replace_probability):
        uwu_str = "uwu"
        token = re.sub(
            r"u+", uwu_str, token, flags=re.IGNORECASE, count=random.choice(range(0, 2))
        )

    return token


def fuckyer(token: str) -> str:
    extra_fun = ""
    y_choice_1 = ("y" if decision(0.5) else "i") * random.choice(range(1, 5))
    y_choice_2 = ("y" if decision(0.5) else "i") * random.choice(range(1, 5))
    if decision(0.5):
        extra_fun = f"w{'u' * random.choice(range(1, 5))}k{y_choice_2}"
    token = re.sub(
        r"([Ff])?uck(er|ing)?",
        lambda match: f"{match.group(1) or ''}{'u'*random.choice(range(1,5))}k{y_choice_1}{match.group(2) or ''}"
        + " "
        + extra_fun,
        token,
    )
    return token


def garbage(token: str) -> str:
    # inserting gay
    token = re.sub(r"([a-fh-zA-FH-Z])a+y+", lambda match: f"{match.group(1)}gay", token)

    # hewwo
    token = re.sub(r"([Hh])e+ll+o+?", lambda match: f"{match.group(1)}ewwo", token)

    # cute -> koot
    token = re.sub(
        r"([Cc])u+te",
        lambda match: f"{match.group(1)}oo{'o'*random.randint(0,5)}t",
        token,
    )

    # as -> ass asss
    if decision(0.5):
        token = re.sub(
            r"([aA])([sS])($|[^s])",
            lambda match: f"{match.group(1)}{match.group(2)*random.randint(2,3)}t",
            token,
        )

    # TODO: refactor (me -> meh|me -> meow) together?
    # me -> meow
    if decision(me_2_meow_swap_probability):
        token = re.sub(
            r"^me+$",
            lambda match: f"m{'e'*random.randint(1,3)}{'o'*random.randint(1,3)}w",
            token,
            flags=re.IGNORECASE,
        )

    # me -> meh
    if decision(me_2_meh_swap_probability):
        token = re.sub(
            r"^me+$",
            lambda match: f"m{'e' * random.randint(1, 3)}h",
            token,
            flags=re.IGNORECASE,
        )

    # my -> mah, myah
    if decision(0.5):
        token = re.sub(
            r"^my+$",
            lambda match: f"m{'y' if decision(0.3) else ''}{'a'*random.randint(2,3)}{'h' if decision(0.5) else ''}",
            token,
        )

    # ion -> shun
    if decision(0.5):
        token = re.sub(r"ion$", "shun", token)

    # .ome -> .um
    if decision(0.5):
        token = re.sub(r"([a-zA-Z])ome", lambda match: f"{match.group(1)}um", token)

    # teh or da
    if decision(0.5):
        token = re.sub(r"^([Tt])he$", lambda match: f"{match.group(1)}eh", token)
    else:
        token = re.sub(
            r"^([Tt])he$",
            lambda match: f"{'D' if match.group(1) == 'T' else 'd'}a",
            token,
        )

    # ing -> inn
    if decision(0.5):
        token = re.sub(
            r"ing$",
            f"in{'n'*random.randint(0,4) if decision(0.5) else 'in'*random.randint(0,4) }",
            token,
            flags=re.IGNORECASE,
        )

    # ks -> ksksksk
    if decision(ksks_enlargement_probability):
        token = re.sub(
            r"[kK][sS]|[sS][kK]",
            lambda match: f"{match.group(0)*random.randint(2,6)}",
            token,
            flags=re.IGNORECASE,
        )

    # uck -> ucc, uccci
    if decision(uck_to_ucc_swap_probability):
        token = re.sub(
            r"u+c+k+",
            lambda match: f"u{'c'*random.randint(2,6)}{'i'*random.randint(0,3)}",
            token,
            flags=re.IGNORECASE,
        )

    if decision(sub_to_subby_swap_probability):
        token = re.sub(
            r"s(u+)b",
            lambda match: f"s{match.group(1)}bb{('y' if decision(0.5) else 'i')*random.randint(1,2)}",
            token,
            flags=re.IGNORECASE,
        )
    return token


def reeeer(token: str) -> str:
    if decision(REEE_probability):
        token = re.sub(
            r"([Rr])e*",
            lambda match: f"{match.group(1)}e" + "e" * random.choice(range(1, 15)),
            token,
        )
        if decision(REEE_allcaps_probability):
            token = token.upper()
    return token


def rawrer(token: str) -> str:
    token = re.sub(r"ra([a-zA-Z])?", lambda match: f"rawr{match.group(1) or ''}", token)
    token = re.sub(
        r"ar([a-zA-Z])?", lambda match: f"arawr{match.group(1) or ''}", token
    )
    token = re.sub(r"([Rr])oar", lambda match: f"{match.group(1)}awr", token)

    return token


def jizzer(token: str) -> str:
    token = re.sub(r"(.iz+)", "jizz", token)
    return token


# TODO: reval this some of it is ineffective i think
def cummer(token: str) -> str:
    token = re.sub(r"(.ome|co+m|co+n{1,3})", "cum", token)
    token = re.sub(r"(c.{0,2}u+m)", "cum", token)
    token = re.sub(r"(cau|cou)", "cum", token)
    token = re.sub(r"(cow)", "cum", token)
    token = re.sub(r"(son|sun$)", "cum", token)

    token = re.sub(r"([a-bd-zA-BD-Z])um", lambda match: f"{match.group(1)}cum", token)

    token = re.sub(
        r"([a-bd-zA-BD-Z])u(nn|mm)([yi])",
        lambda match: f"{match.group(1)}cumm{match.group(3)}",
        token,
    )
    token = re.sub(r"(cally)", "cummy", token)

    return token


garbage_tokens = [
    "omega",
    "UWU",
    "uwu",
    "OWO",
    "owo",
    "X.X",
    "ksk ksk ksk ksk",
    "x.x",
    "nyah",
    "nyahhhh",
    "uguu",
    "oof",
    "ooomf",
    "UGUU",
    "3==D",
    ":p",
    ">:p",
    ":3",
    ">:3",
    ">D3",
    ">B3",
    "num",
    "nom",
    "nuzzles",
    "huehue",
    "LOL",
    "lol",
    "lel",
    "X3",
    "xd",
    "XD",
    "sexnumber",
    "69",
    "420",
    "weednumber",
]


def add_random_garbage_token():
    return random.choice(garbage_tokens)


with open("emoji.csv", newline="\n", encoding="utf-8") as csvfile:
    text_face_emojis = list(
        [item for sublist in csv.reader(csvfile) for item in sublist]
    )

con = sqlite3.connect(":memory:")
with open(
    "emoji-sentiment-data\\Emoji_Sentiment_Data_v1.0.csv",
    newline="\n",
    encoding="utf-8",
) as csvfile:
    df = pandas.read_csv(csvfile)
    df.to_sql("Emoji_Sentiment_Data", con, if_exists="append", index=False)


with open("simple_text_emoji.csv", newline="\n", encoding="utf-8") as csvfile:
    simple_text_emojis = list(
        [item for sublist in csv.reader(csvfile) for item in sublist]
    )


with open("action_verbs.csv", newline="\n", encoding="utf-8") as csvfile:
    action_verbs = list([item for sublist in csv.reader(csvfile) for item in sublist])


with open("rp_pronouns.csv", newline="\n", encoding="utf-8") as csvfile:
    rp_pronouns = list([item for sublist in csv.reader(csvfile) for item in sublist])


def get_random_text_face_emojis():
    return random.choice(text_face_emojis)


def get_random_simple_text_emojis():
    return random.choice(simple_text_emojis)


def decision(probability) -> bool:
    return random.random() < probability


def generate_spongebob_text(token: str) -> str:
    """gEnErAtEs sPoNgEbOb mEmE TeXt"""
    spongebob_text = ""
    for i, char in enumerate(token):
        if i % 2 == 0:
            spongebob_text += char.lower()
        else:
            spongebob_text += char.upper()
    return spongebob_text


def shuffle_str(token: str) -> str:
    token_str_list = list(token)
    random.shuffle(token_str_list)
    return "".join(token_str_list)


get_rhymes_probability = 0.01
max_runon_rhymes = 3
min_runon_rhymes = 1


def get_runon_of_rhymes(
    token: str,
    min_runon: int = 1,
    max_runon: int = 3,
    max_rhyme_dups: int = 0,
    allow_token_dupe: bool = False,
) -> Set[str]:
    selected_rhymes = set()
    tried_nltk = False
    tried_pronouncing = False
    while True:
        if decision(0.5):
            tried_pronouncing = True
            rhymes = get_pronouncing_rhyme(token)
            if not allow_token_dupe:
                try:
                    rhymes.remove(token)
                except ValueError:
                    pass
            if rhymes:
                selected_rhymes.add(random.choice(rhymes))
        else:
            tried_nltk = True
            level = 4
            while True:
                rhymes = get_nltk_rymes(token, level)
                if not allow_token_dupe:
                    try:
                        rhymes.remove(token)
                    except ValueError:
                        pass
                if rhymes:
                    selected_rhymes.add(random.choice(rhymes))
                    break
                if level == 0:
                    break
                level -= 1

        if (decision(0.5) and len(selected_rhymes) == min_runon) or len(
            selected_rhymes
        ) == max_runon:
            break

        if not selected_rhymes and tried_pronouncing and tried_nltk:
            break
    return selected_rhymes


def get_pronouncing_rhyme(token: str) -> List[str]:
    return pronouncing.rhymes(token)


def get_nltk_rymes(token: str, level) -> List[str]:
    # TODO: stub
    def rhyme(inp, level):
        """
        1 bad rhymes
        2
        4 good rhymes
        """
        entries = nltk.corpus.cmudict.entries()
        syllables = [(word, syl) for word, syl in entries if word == inp]
        rhymes = []
        for (word, syllable) in syllables:
            rhymes += [
                word for word, pron in entries if pron[-level:] == syllable[-level:]
            ]
        return set(rhymes)

    return list(rhyme(token, level))


def over_emphasise_punctuation(token: str, max_fuck: int = 4) -> str:
    if token == "?":
        token += "".join(
            random.choices(
                [
                    "1",
                    # "i",
                    "!",
                    "?",
                    # "I",
                    # "/",
                    # ".",
                    # "\\"
                ],
                k=random.choice(range(0, max_fuck)),
            )
        )
        token = shuffle_str(token)
    if token == "!":
        token += "".join(
            random.choices(
                [
                    "1",
                    # "i",
                    "!",
                    "?",
                    # "I",
                    # "/",
                    "|",
                ],
                k=random.choice(range(0, max_fuck)),
            )
        )
        token = shuffle_str(token)
    if token == ".":
        token += "".join(
            random.choices([",", "."], k=random.choice(range(0, max_fuck)))
        )
        token = shuffle_str(token)

    return token


def to_rp_text(token: str) -> str:
    return f"*{token}*"


def get_random_action_verb():
    return random.choice(action_verbs)


def get_random_rp_pronoun():
    return random.choice(rp_pronouns)


def random_swap_char(token: str, swaps_percent: float = 0.2) -> str:
    if len(token) < 3:  # dont do this for small tokens as they become un decipherable
        return token
    swaps = int(ceil(len(token) * swaps_percent))
    indexes = random.choices(range(len(token)), k=swaps)
    for i in indexes:
        token = "".join(
            [
                token[w] if w != i else random.choice(string.ascii_letters)
                for w in range(len(token))
            ]
        )
    return token


def random_insert_char(token: str, insert_percent: float = 0.1) -> str:
    swaps = int(ceil(len(token) * insert_percent))
    indexes = random.choices(range(len(token)), k=swaps)
    token_str_list = list(token)
    for i in indexes:
        token_str_list.insert(i, random.choice(string.ascii_letters))
    token = "".join(token_str_list)
    return token


def token_to_leet(token: str) -> str:
    if len(token) < 5:  # leet speaking small text has hard to read results
        return token
    leet_char_mapping = {
        # "a": "4",
        "a": "@",
        "e": "3",
        "8": "&",
        "l": "1",
        "o": "0",
        "s": "5",
        "i": "1",
    }
    getchar = (
        lambda c: leet_char_mapping[c.lower()] if c.lower() in leet_char_mapping else c
    )
    return "".join(getchar(c) for c in token)


# TODO: lots of options maybe something learned?
def utf_8_char_swaps(token: str) -> str:
    if decision(0.5):
        token = re.sub(r"ae", "æ", token)
        token = re.sub(r"AE", "Æ", token)
    if decision(0.3):
        token = re.sub(r"ea", "æ", token)
        token = re.sub(r"EA", "Æ", token)
    return token


def get_token_random_definition(token: str) -> Optional[str]:
    synsets = wn.synsets(token)
    if synsets:
        return random.choice(synsets).definition()


def fuck_sentence(sentance: Sentence) -> List[str]:
    new_tokens = []
    # TODO: determine mood classifier for sentence and add respective emoji

    for token in sentance.tokens:
        original_token = token

        if decision(random_synonym_probability):
            token = replace_with_random_synonym(token)
        if decision(censor_profanity_probability) and profanity.contains_profanity(
            token
        ):
            token = custom_censoring(token, censor_profanity_percent)
        elif decision(random_censor_probability):
            token = custom_censoring(token, random_censor_percent)

        if re.match("musk", token, flags=re.IGNORECASE):
            add_husky = True
        else:
            add_husky = False

        # processing
        fucked_token = fuck_token(token)

        # post processing
        new_tokens.append(fucked_token)

        if decision(add_definition_in_parenthesis_probability):
            definiton = get_token_random_definition(token)
            if definiton:
                new_tokens += [
                    f"[[{fuck_token('DEFINITION')} {token.upper()}:",
                    f"{fuck_text_blob(definiton)}]]",
                ]

        if add_husky:
            new_tokens.append(fuck_token("husky"))

        if add_random_garbage and decision(add_random_garbage_probability):
            new_tokens.append(fuck_token(add_random_garbage_token()))
        if add_randomly_text_face_emoji and decision(
            add_randomly_text_face_emojis_probability
        ):
            new_tokens.append(get_random_text_face_emojis())
        if add_random_simple_text_emoji and decision(
            # TODO: use textblob to determine mood of text and insert faces accordingly
            #   likely need to do this after reconstruction of the text blob and go through this
            #   sentance by sentance rather than word by word
            add_random_simple_text_emoji_probability
        ):
            new_tokens.append(get_random_simple_text_emojis())
        if add_random_rp_action and decision(
            add_random_rp_mid_sentance_action_probability
        ):
            new_tokens.append(get_random_rp_action_sentence())
    if add_random_rp_action and decision(add_random_rp_end_sentance_action_probability):
        new_tokens.append(get_random_rp_action_sentence())

    if decision(random_lorem_ipsum_probabilty):
        new_tokens.append(get_random_lorem_ipsum())
    return new_tokens


def add_ending_y(token: str) -> str:
    return re.sub(r"([a-zA-Z]{4,}[^sy])", lambda match: f"{match.group(1)}y", token)


def get_random_lorem_ipsum() -> str:
    """get lorem ipsum sentence"""
    if decision(lorem_ipsum_fuck_probabilty):
        fuck_sentence(Sentence(lorem.sentence()))
    return lorem.sentence()


def fuck_token(token: str) -> str:
    # TODO: determine mood classifier for token and add respective emoji
    if decision(split_compound_word_probability):
        tokens = split_compound_word(token)
    else:
        tokens = [token]

    # TODO: migrate fuck_token to maybe a generator?
    fucked_tokens = []
    for token in tokens:
        if decision(lazy_char_subbing_probability):
            token = lazy_char_subbing(token)
        fucked_token = knotter(fuckyer(reeeer(rawrer(garbage(owoer(cummer(token)))))))

        if decision(add_extra_ed_probability):
            fucked_token = add_extra_ed(fucked_token)

        if decision(random_ending_y_probabilty):
            fucked_token = add_ending_y(fucked_token)

        if add_random_plurals and decision(add_random_plurals_probability):
            for word in TextBlob(fucked_token).words:
                fucked_token = Word(word).pluralize()

        if randomly_lemmatize and decision(randomly_lemmatize_probability):
            for word in TextBlob(fucked_token).words:
                fucked_token = Word(word).lemmatize()

        if randomly_capitalize_word and decision(randomly_capitalize_word_probability):
            fucked_token = fucked_token.upper()

        if randomly_spongebob_word and decision(randomly_spongebob_word_probability):
            fucked_token = generate_spongebob_text(fucked_token)

        if randomly_overemphasis_punct and decision(
            randomly_overemphasis_punct_probability
        ):
            fucked_token = over_emphasise_punctuation(
                fucked_token, randomly_overemphasis_punct_max_fuck
            )

        if decision(common_mispellings_probability):
            fucked_token = common_mispellings(fucked_token)

        if randomly_swap_char and decision(randomly_swap_char_probability):
            fucked_token = random_swap_char(
                fucked_token, randomly_swap_char_swap_percent
            )

        if randomly_insert_char and decision(randomly_insert_char_probability):
            fucked_token = random_insert_char(
                fucked_token, randomly_insert_char_insert_percent
            )
        if decision(utf_8_char_swaps_probability):
            fucked_token = utf_8_char_swaps(fucked_token)

        if random_leet_speek and decision(random_leet_speek_probability):
            fucked_token = token_to_leet(fucked_token)

        if decision(common_mispellings_probability):
            fucked_token = common_mispellings(fucked_token)

        # TODO: discord format options
        if decision(bold_text_probability):
            fucked_token = bold_text(fucked_token)
        elif decision(back_tick_text_probability):
            fucked_token = back_tick_text(fucked_token)

        fucked_tokens.append(fucked_token)

        if decision(add_x3_if_token_has_rawr_probability) and (
            "rawr" in fucked_token.lower()
        ):
            fucked_tokens.append("X3" if decision(0.5) else "x3")

        if decision(adding_ending_ksksk_andioop_probability) and (
            fucked_token.lower().endswith("ksk")
            or fucked_token.lower().endswith("sks")
            or "ksksk" in fucked_token.lower()
            or "sksks" in fucked_token.lower()
        ):
            for i in range(random.randint(1, 2)):
                fucked_tokens.append(fuck_token("andioop"))
        if decision(adding_ending_ksksk_save_the_turtles_probability) and (
            fucked_token.lower().endswith("ksk")
            or fucked_token.lower().endswith("sks")
            or "ksksk" in fucked_token.lower()
            or "sksks" in fucked_token.lower()
        ):
            fucked_tokens.append(fuck_text_blob("save the turtles!"))

        if decision(get_rhymes_probability):
            for rhyme in get_runon_of_rhymes(
                token, max_runon=max_runon_rhymes, min_runon=min_runon_rhymes
            ):
                fucked_rhyme = fuck_token(rhyme)
                print(f"adding rhyme {token} {rhyme} {fucked_rhyme}")
                fucked_tokens.append(fucked_rhyme)

    return " ".join(fucked_tokens)


def bold_text(token: str) -> str:
    if not token.strip(
        string.punctuation
    ):  # don't bold tokens of all punctuation as it bugs up rejoining punctuation later *todo: maybe alternate fix?
        return token
    return f"**{token.strip('*')}**"


def get_random_rp_action_sentence() -> str:
    more_verbs = []
    more_verbs_probability = 1
    while True:
        if decision(more_verbs_probability):
            additional_verb = get_random_action_verb()
            if decision(0.5):  # TODO: config
                additional_verb = Word(additional_verb).lemmatize()
            additional_verb = fuck_token(additional_verb)
            additional_verb = Word(additional_verb).pluralize()
            more_verbs.append(additional_verb)
        else:
            break
        more_verbs_probability -= more_verbs_probability_decay

    noun = get_random_rp_pronoun()
    if decision(0.5):  # TODO: config
        noun = Word(noun).lemmatize()

    # TODO: add boolean for enable
    noun = fuck_token(noun)
    noun = Word(noun).pluralize()
    return to_rp_text(f"{' and '.join(more_verbs)}{' ' if more_verbs else ''}{noun}")


def lazy_char_subbing(token: str) -> str:
    """e.g.you -> u are -> r"""
    # TODO: better capital replacement

    # you -> u, yuu
    token = re.sub(
        "^y+(o+)?u+$",
        lambda match: f"u" if decision(0.5) else f"y{'u'*random.randint(1,4)}",
        token,
        flags=re.IGNORECASE,
    )

    # are -> r, arrr
    token = re.sub(
        "^a+(r+)?e+$",
        lambda match: f"r" if decision(0.5) else f"a{'r' * random.randint(1, 4)}",
        token,
        flags=re.IGNORECASE,
    )

    # with -> wif
    token = re.sub(
        "^wi+th+$",
        lambda match: f"w{'i' * random.randint(1, 4)}{'f' * random.randint(1, 4)}",
        token,
        flags=re.IGNORECASE,
    )

    # what -> wat
    token = re.sub(
        "^wha+t$",
        lambda match: f"w{'a' * random.randint(1, 4)}t",
        token,
        flags=re.IGNORECASE,
    )

    # er -> ur
    token = re.sub(
        "er",
        lambda match: f"ur",
        token,
        flags=re.IGNORECASE,
        count=random.randint(0, 2),
    )

    # easy -> ez
    token = re.sub(
        "^ea+s+y+$",
        lambda match: f"e{'z' * random.randint(1, 3)}",
        token,
        flags=re.IGNORECASE,
    )

    # to,too, -> 2
    token = re.sub("to+$", lambda match: f"2", token, flags=re.IGNORECASE,)
    return token


# TODO: funny -> funni spells -> spellz
def common_mispellings(token: str) -> str:
    # TODO: cleanup
    token = re.sub(
        "([^\s])y$", lambda match: f"{match.group(1)}{'i'*random.randint(1,1)}", token
    )
    token = re.sub(
        "([^\s])Y$", lambda match: f"{match.group(1)}{'Y'*random.randint(1,2)}", token
    )
    token = re.sub(
        "([^\s])s$", lambda match: f"{match.group(1)}{'z'*random.randint(1,2)}", token
    )
    token = re.sub(
        "([^\s])S$", lambda match: f"{match.group(1)}{'Z'*random.randint(1,2)}", token
    )
    token = re.sub(
        "([^\s])z$", lambda match: f"{match.group(1)}{'s'*random.randint(1,2)}", token
    )
    token = re.sub(
        "([^\s])Z$", lambda match: f"{match.group(1)}{'S'*random.randint(1,2)}", token
    )
    token = re.sub(
        "([eE])([iI])", lambda match: f"{match.group(2)}{match.group(1)}", token
    )
    return token


def fix_punctuation_spacing(text: str) -> str:
    # TODO: this is a meh way to solve punct being incorrectly joined should investigate
    return re.sub(
        "([^\s]) ([!\?.,]+)", lambda match: f"{match.group(1)}{match.group(2)}", text
    )


back_tick_text_probability = 0.05


def back_tick_text(token: str) -> str:
    if not token.strip(
        string.punctuation
    ):  # don't back_tick tokens of all punctuation as it bugs up rejoining punctuation later *todo: maybe alternate fix?
        return token
    return f"`{token.strip('`')}`"


def split_compound_word(token: str) -> List[str]:
    tokens = splitter.split(str(token))
    if isinstance(tokens, list):
        return tokens
    return [tokens]


def add_extra_ed(token: str) -> str:
    return re.sub(
        "([a-zA-Z]{2,})(d|ed)$",
        lambda match: f"{match.group(1)}{'ed' * random.randint(1, 2)}",
        token,
        flags=re.IGNORECASE,
    )


# TODO: grabagey code duplication
def custom_censoring(swear_word: str, censor_percent: float = 0.25) -> str:
    if len(swear_word) <= 3:  # dont censor stuff this small
        return swear_word
    censor_word_list = list("@#$%*")
    swaps = int(ceil(len(swear_word) * censor_percent))
    indexes = random.choices(range(1, len(swear_word)), k=swaps)
    for i in indexes:
        swear_word = "".join(
            [
                swear_word[w] if w != i else random.choice(censor_word_list)
                for w in range(len(swear_word))
            ]
        )
    return swear_word


def replace_with_random_synonym(token: str) -> str:
    # TODO: fill in with all synonyms for lulz?
    # TODO: download manual dictionary
    return token


def fuck_text_blob(text: str) -> str:
    new_tokens = []

    # TODO: go sentance by sentance token by token all for sentiment analysis
    for sentence in TextBlob(text).sentences:
        new_tokens += fuck_sentence(sentence)

    out_str = TreebankWordDetokenizer().detokenize(new_tokens)
    out_str = fix_punctuation_spacing(out_str)
    return out_str


def main():
    text = """reproduce Hey there guys today I'm making a cool python script that recompiles text into a more unreadable heap of garbage. Let me know what you guys think of it!"""
    # text = 'Oop why did it double post that fuck'
    print(f"before: {text}")
    out_str = fuck_text_blob(text)
    print()
    print(f"after:  {out_str}")


if __name__ == "__main__":
    main()
