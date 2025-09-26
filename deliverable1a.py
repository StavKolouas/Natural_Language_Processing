import re
import nltk



pattern = r"\b\w+\b"
first_sentence = "Hope you too, to enjoy it as my deepest wishes."
second_sentence = "Because I didn't see that part final yet, or maybe I missed, I apologize if so."


tokens1 = re.findall(pattern, first_sentence)
tokens2 = re.findall(pattern, second_sentence)

print(tokens1)
print(tokens2)

#custom grammar for parsing the sentences as they are written
grammar = nltk.CFG.fromstring("""
S -> VP NP | NP VP
NP -> N PP Det Adj N | N Adv PP | IN N | Det N Adj Adv IN Adv N | N | IN Adv
VP -> V NP | V | V NP VP | VBD RB V NP VP  
Det -> 'my' | 'that'
Adj -> 'deepest' | 'final'
N -> 'you' | 'it' | 'wishes' | 'I' | 'part'
V -> 'enjoy' | 'Hope' | 'see' | 'missed' | 'apologize'
Adv -> 'too' | 'yet' | 'maybe' | 'so'
PP -> 'to' | 'as'
IN -> 'Because' | 'if' | 'or'
VBD -> 'didn'
RB -> 't'
""")


parser = nltk.ChartParser(grammar)


def reconstruct(tokens):
    parses = list(parser.parse(tokens))
    
    tree = parses[0]
    print("\n Parse tree:")
    tree.pretty_print()

    terminals = tree.leaves()


    if terminals[0] == "Hope":
        terminals = ["I"] + terminals

    # better words to improve tone
    for i in range(len(terminals) - 3):
        if (terminals[i] == "as" and terminals[i+1] == "my" and
            terminals[i+2] == "deepest" and terminals[i+3] == "wishes"):
            terminals[i] = "with"

    # Rule C: "that part final" → "that part finished"
    for i in range(len(terminals) - 2):
        if terminals[i:i+3] == ["that", "part", "final"]:
            terminals[i+2] = "finished"

    #adverbs shold be at the end
    if "too" in terminals:
        terminals = [t for t in terminals if t != "too"]
        terminals.append("too")

    # combine didn and t again
    i = 0
    while i < len(terminals) - 1:
        if terminals[i] == "didn" and terminals[i+1] == "t":
            terminals[i] = "didn't"
            del terminals[i+1]
        else:
            i += 1

    #first letter of starting word needs to be in capital
    terminals[0] = terminals[0].capitalize()

    # Rebuild sentence
    sentence = " ".join(terminals) + "."
    return sentence


print("Reconstructed:", reconstruct(tokens1))

print("Reconstructed:", reconstruct(tokens2))
