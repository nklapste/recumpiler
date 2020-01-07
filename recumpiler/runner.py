#!/usr/bin/env python
# -*- coding: utf-8 -*-

""""""
import configparser
import random
import re
from typing import List

from better_profanity import profanity
from nltk.tokenize.treebank import TreebankWordDetokenizer
from textblob import Sentence, TextBlob, Word

from recumpiler.mutators import *
from recumpiler.utils import get_default_mutator_config


@logged_mutator
def recumpile_sentence(sentance: Sentence) -> List[str]:
    new_tokens = []
    # TODO: determine mood classifier for sentence and add respective emoji

    for token in sentance.tokens:
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
        recumpiled_token = recumpile_token(token)

        # post processing
        new_tokens.append(recumpiled_token)

        if decision(add_definition_in_parenthesis_probability):
            definition = get_token_random_definition(token)
            if definition:
                new_tokens += [
                    f"[[{recumpile_token('DEFINITION')} {token.upper()}:",
                    f"{recumpile_text(definition)}]]",
                ]

        if add_husky:
            new_tokens.append(recumpile_token("husky"))

        if add_random_garbage and decision(add_random_garbage_probability):
            new_tokens.append(recumpile_token(add_random_garbage_token()))
        if add_randomly_text_face_emoji and decision(
            add_randomly_text_face_emoji_probability
        ):
            new_tokens.append(get_random_text_face_emojis())
        if add_random_simple_text_emoji and decision(
            # TODO: use textblob to determine mood of text and insert faces
            #  accordingly likely need to do this after reconstruction of the
            #  text blob and go through this sentence by sentence rather than
            #  word by word.
            add_random_simple_text_emoji_probability
        ):
            new_tokens.append(get_random_simple_text_emojis())
        if add_random_rp_action and decision(
            add_random_rp_mid_sentence_action_probability
        ):
            new_tokens.append(get_random_rp_action_sentence())
    if add_random_rp_action and decision(add_random_rp_end_sentence_action_probability):
        new_tokens.append(get_random_rp_action_sentence())

    if decision(random_lorem_ipsum_probability):
        new_tokens.append(get_random_lorem_ipsum())
    return new_tokens


@logged_mutator
def recumpile_token(token: str) -> str:
    # TODO: determine mood classifier for token and add respective emoji
    if decision(split_compound_word_probability):
        tokens = split_compound_word(token)
    else:
        tokens = [token]

    # TODO: migrate fuck_token to maybe a generator?
    fucked_tokens = []
    for token in tokens:
        relevant_emoji = None
        if decision(add_text_relevant_emoji_probability):
            relevant_emoji = find_text_relevant_emoji(
                token
            )  # TODO: add ability to get multiple?
            if relevant_emoji and decision(wrap_text_relevant_emoji_probability):
                fucked_tokens.append(relevant_emoji)

        if decision(lazy_char_subbing_probability):
            token = lazy_char_subbing(token)

        # TODO: this is a potential for unexpected behavior
        if decision(word_to_num_probability):
            token = word_to_num(token)
        if decision(num_to_word_probability):
            token = num_to_word(token)

        fucked_token = knotter(fuckyer(reeeer(rawrer(garbage(owoer(cummer(token)))))))

        if decision(add_extra_ed_probability):
            fucked_token = add_extra_ed(fucked_token)

        if decision(random_ending_y_probability):
            fucked_token = add_ending_y(fucked_token)

        # TODO: likey making fu@k into k
        # TODO: NOTE: indeed it is doing this fu@k
        #   >>>list(TextBlob("fu@k").words)
        #   ['fu', 'k']
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

        if randomly_overemphasis_punctuation and decision(
            randomly_overemphasis_punctuation_probability
        ):
            fucked_token = over_emphasise_punctuation(
                fucked_token, randomly_overemphasis_punctuation_max_fuck
            )

        if decision(common_misspellings_probability):
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

        if random_leet_speak and decision(random_leet_speak_probability):
            fucked_token = token_to_leet(fucked_token)

        if decision(common_misspellings_probability):
            fucked_token = common_mispellings(fucked_token)

        # TODO: likely also breaking the spacing between punctuation kittly 1!
        # TODO: `fucked` went to `DS` investigate
        # TODO: likely this is at fault
        if decision(homofiy_probability):
            fucked_token = homoify(fucked_token, homofiy_probability)

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
                fucked_tokens.append(recumpile_token("andioop"))
        if decision(adding_ending_ksksk_save_the_turtles_probability) and (
            fucked_token.lower().endswith("ksk")
            or fucked_token.lower().endswith("sks")
            or "ksksk" in fucked_token.lower()
            or "sksks" in fucked_token.lower()
        ):
            fucked_tokens.append(recumpile_text("save the turtles!"))

        if decision(fucking_normies_addition) and "reee" in fucked_token.lower():
            fucked_tokens.append(recumpile_text("fucking normies!"))

        if decision(get_rhymes_probability):
            for rhyme in get_runon_of_rhymes(
                token, max_runon=max_runon_rhymes, min_runon=min_runon_rhymes
            ):
                fucked_rhyme = recumpile_token(rhyme)
                fucked_tokens.append(fucked_rhyme)

        if relevant_emoji:
            fucked_tokens.append(relevant_emoji)

    for i, fucked_token in enumerate(fucked_tokens):
        if decision(space_gap_text_probability):
            # TODO: this modification may be better placed elsewhere
            fucked_token = space_gap_text(
                fucked_token,
                min_gap_size=space_gap_text_min_gap_size,
                max_gap_size=space_gap_text_max_gap_size,
            )
        # TODO: discord format options
        if decision(bold_text_probability):
            fucked_token = bold_text(fucked_token)
        elif decision(back_tick_text_probability):
            fucked_token = back_tick_text(fucked_token)
        fucked_tokens[i] = fucked_token

    return " ".join(fucked_tokens)


@logged_mutator
def recumpile_line(text: str) -> str:
    new_tokens = []
    for sentence in TextBlob(text).sentences:
        new_tokens += recumpile_sentence(sentence)
    out_str = TreebankWordDetokenizer().detokenize(new_tokens)
    out_str = fix_punctuation_spacing(out_str)

    return out_str


@logged_mutator
def recumpile_text(
    text: str, mutator_config: configparser.ConfigParser = get_default_mutator_config()
) -> str:
    # TODO: preserve spacing better / Maybe use nltk tokenizers instead of a
    #  split method
    # TODO: go sentence by sentence token by token all for sentiment analysis
    lines = []

    for line in text.split("\n"):
        lines.append(recumpile_line(line))
    return "\n".join(lines)
