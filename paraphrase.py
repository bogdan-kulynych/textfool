import attr
import nltk
import spacy

from collections import OrderedDict
from functools import partial

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
from pywsd.lesk import simple_lesk as disambiguate

from typos import typos


nlp = spacy.load('en')


# Penn TreeBank POS tags:
# http://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
supported_pos_tags = [
    # 'CC',   # coordinating conjunction
    # 'CD',   # Cardinal number
    # 'DT',   # Determiner
    # 'EX',   # Existential there
    # 'FW',   # Foreign word
    # 'IN',   # Preposition or subordinating conjunction
    'JJ',   # Adjective
    # 'JJR',  # Adjective, comparative
    # 'JJS',  # Adjective, superlative
    # 'LS',   # List item marker
    # 'MD',   # Modal
    'NN',   # Noun, singular or mass
    'NNS',  # Noun, plural
    'NNP',  # Proper noun, singular
    'NNPS', # Proper noun, plural
    # 'PDT',  # Predeterminer
    # 'POS',  # Possessive ending
    # 'PRP',  # Personal pronoun
    # 'PRP$', # Possessive pronoun
    'RB',   # Adverb
    # 'RBR',  # Adverb, comparative
    # 'RBS',  # Adverb, superlative
    # 'RP',   # Particle
    # 'SYM',  # Symbol
    # 'TO',   # to
    # 'UH',   # Interjection
    'VB',   # Verb, base form
    'VBD',  # Verb, past tense
    'VBG',  # Verb, gerund or present participle
    'VBN',  # Verb, past participle
    'VBP',  # Verb, non-3rd person singular present
    'VBZ',  # Verb, 3rd person singular present
    # 'WDT',  # Wh-determiner
    # 'WP',   # Wh-pronoun
    # 'WP$',  # Possessive wh-pronoun
    # 'WRB',  # Wh-adverb
]


@attr.s
class SubstitutionCandidate:
    token_position = attr.ib()
    similarity_rank = attr.ib()
    original_token = attr.ib()
    candidate_word = attr.ib()


def vsm_similarity(doc, original, synonym):
    window_size = 3
    start = max(0, original.i - window_size)
    return doc[start: original.i + window_size].similarity(synonym)


def _get_wordnet_pos(spacy_token):
    '''Wordnet POS tag'''
    pos = spacy_token.tag_[0].lower()
    if pos in ['a', 'n', 'v']:
        return pos


def _synonym_prefilter_fn(token, synonym):
    '''
    Similarity heuristics go here
    '''
    if  (len(synonym.text.split()) > 2) or \
        (synonym.lemma == token.lemma) or \
        (synonym.tag != token.tag) or \
        (token.text.lower() == 'be'):
        return False
    else:
        return True


def _generate_synonym_candidates(doc, disambiguate=True, rank_fn=None):
    '''
    Generate synonym candidates.

    For each token in the doc, the list of WordNet synonyms is expanded.
    the synonyms are then ranked by their GloVe similarity to the original
    token and a context window around the token.

    :param disambiguate: Whether to use lesk sense disambiguation before
            expanding the synonyms.
    :param rank_fn: Functions that takes (doc, original_token, synonym) and
            returns a similarity score
    '''
    if rank_fn is None:
        rank_fn=vsm_similarity

    candidates = []
    for position, token in enumerate(doc):
        if token.tag_ in supported_pos_tags:
            wordnet_pos = _get_wordnet_pos(token)
            wordnet_synonyms = []
            if disambiguate:
                try:
                    synset = disambiguate(
                           doc.text, token.text, pos=wordnet_pos)
                    wordnet_synonyms = synset.lemmas()
                except:
                    continue
            else:
                synsets = wn.synsets(token.text, pos=wordnet_pos)
                for synset in synsets:
                    wordnet_synonyms.extend(synset.lemmas())

            synonyms = []
            for wordnet_synonym in wordnet_synonyms:
                spacy_synonym = nlp(wordnet_synonym.name().replace('_', ' '))[0]
                synonyms.append(spacy_synonym)

            synonyms = filter(partial(_synonym_prefilter_fn, token),
                              synonyms)
            synonyms = reversed(sorted(synonyms,
                                key=partial(rank_fn, doc, token)))

            for rank, synonym in enumerate(synonyms):
                candidate_word = synonym.text
                candidate = SubstitutionCandidate(
                        token_position=position,
                        similarity_rank=rank,
                        original_token=token,
                        candidate_word=candidate_word)
                candidates.append(candidate)

        return candidates


def _generate_typo_candidates(doc, min_token_length=4, rank=1000):
    candidates = []
    for position, token in enumerate(doc):
        if (len(token)) < min_token_length:
            continue

        for typo in typos(token.text):
            candidate = SubstitutionCandidate(
                    token_position=position,
                    similarity_rank=rank,
                    original_token=token,
                    candidate_word=typo)
            candidates.append(candidate)

    return candidates


def _compile_perturbed_tokens(doc, accepted_candidates):
    '''
    Traverse the list of accepted candidates and do the token substitutions.
    '''
    candidate_by_position = {}
    for candidate in accepted_candidates:
        candidate_by_position[candidate.token_position] = candidate

    final_tokens = []
    for position, token in enumerate(doc):
        word = token.text
        if position in candidate_by_position:
            candidate = candidate_by_position[position]
            word = candidate.candidate_word.replace('_', ' ')
        final_tokens.append(word)

    return final_tokens


def perturb_text(
        doc,
        add_typos=True,
        rank_fn=None,
        heuristic_fn=None,
        halt_condition_fn=None,
        verbose=False):
    '''
    Perturb the text by replacing some words with their WordNet synonyms,
    sorting by GloVe similarity between the synonym and the original context
    window, and optional heuristic.

    :param doc: Document to perturb.
    :type doc: spacy.tokens.doc.Doc
    :param rank_fn: See `_generate_synonym_candidates``.
    :param heuristic_fn: Ranks the best synonyms using the heuristic.
            If the value of the heuristic is negative, the candidate
            substitution is rejected.
    :param halt_condition_fn: Returns true when the perturbation is
            satisfactory enough.
    :param verbose:

    '''

    heuristic_fn = heuristic_fn or (lambda _, candidate: candidate.similarity_rank)
    halt_condition_fn = halt_condition_fn or (lambda perturbed_text: False)
    candidates = _generate_synonym_candidates(doc, rank_fn=rank_fn)
    if add_typos:
        candidates.extend(_generate_typo_candidates(doc))

    perturbed_positions = set()
    accepted_candidates = []
    perturbed_text = doc.text
    if verbose:
        print('Got {} candidates'.format(len(candidates)))

    sorted_candidates = zip(
            map(partial(heuristic_fn, perturbed_text), candidates),
            candidates)
    sorted_candidates = list(sorted(sorted_candidates,
            key=lambda t: t[0]))

    while len(sorted_candidates) > 0 and not halt_condition_fn(perturbed_text):
        score, candidate = sorted_candidates.pop()
        if score < 0:
            continue
        if candidate.token_position not in perturbed_positions:
            perturbed_positions.add(candidate.token_position)
            accepted_candidates.append(candidate)
            if verbose:
                print('Candidate:', candidate)
                print('Candidate score:', heuristic_fn(perturbed_text, candidate))
                print('Candidate accepted.')
            perturbed_text = ' '.join(
                    _compile_perturbed_tokens(doc, accepted_candidates))

            if len(sorted_candidates) > 0:
                _, candidates = zip(*sorted_candidates)
                sorted_candidates = zip(
                        map(partial(heuristic_fn, perturbed_text),
                            candidates),
                        candidates)
                sorted_candidates = list(sorted(sorted_candidates,
                        key=lambda t: t[0]))
    return perturbed_text


if __name__ == '__main__':
    texts = [
        "Human understanding of nutrition for animals is improving. *Except* for the human animal. If only nutritionists thought humans were animals.",
        "Theory: a climate change denialist has no more inherent right to a media platform than someone who insists the moon may be made of cheese.",
        "Soft skills like sharing and negotiating will be crucial. He says the modern workplace, where people move between different roles and projects, closely resembles pre-school classrooms, where we learn social skills such as empathy and cooperation. Deming has mapped the changing needs of employers and identified key skills that will be required to thrive in the job market of the near future. Along with those soft skills, mathematical ability will be enormously beneficial."
    ]

    def print_paraphrase(text):
        print('Original text:', text)
        doc = nlp(text)
        perturbed_text = perturb_text(doc, verbose=True)
        print('Perturbed text:', perturbed_text)

    for text in texts:
        print_paraphrase(text)
