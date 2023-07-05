from simalign import SentenceAligner

# making an instance of our model.
# You can specify the embedding model and all alignment settings in the constructor.
myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

# The source and target sentences should be tokenized to words.
src_sentence = ["This", "is", "a", "test", "."]
trg_sentence = ["Das", "ist", "ein", "Test", "."]

# The output is a dictionary with different matching methods.
# Each method has a list of pairs indicating the indexes of aligned words (The alignments are zero-indexed).
alignments = myaligner.get_word_aligns(src_sentence, trg_sentence)

for matching_method in alignments:
    print(matching_method, ":", alignments[matching_method])

# Expected output:
# mwmf (Match): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
# inter (ArgMax): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
# itermax (IterMax): [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]