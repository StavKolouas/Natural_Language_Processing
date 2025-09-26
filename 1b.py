import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


text1 = """Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. I am very appreciated the full support of the professor, for our Springer proceedings publication."""
text2 = """During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. Because I didn’t see that part final yet, or maybe I missed, I apologize if so. Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."""


#pipeline 1: Rule-based Automaton
def pipeline1_rule_based(text: str) -> str:
    rules = [
        (r"Hope you too,? to enjoy it", "I hope you enjoy it too"),
        (r"Thank your message", "Thank you for your message"),
        (r"I am very appreciated", "I really appreciate")
    ]
    new_text = text
    for pattern, replacement in rules:
        new_text = re.sub(pattern, replacement, new_text, flags=re.IGNORECASE)
    return new_text


#pipeline 2: TF-IDF cosine similarity-based
def pipeline2_tfidf(text: str) -> str:
    #generate simple candidate rephrases locally
    candidates = [
        text,
        text.replace("bit delay", "slight delay"),
        text.replace("less communication", "reduced communication"),
        re.sub(r"\s+", " ", text)  # normalize whitespace
    ]
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(candidates)
    sim = cosine_similarity(tfidf[0:1], tfidf[1:])
    best_idx = np.argmax(sim)
    return candidates[best_idx + 1]

#pipeline 3: Local synonym/regex-based simple reformation
def pipeline3_local(text: str) -> str:
    synonyms = {
        "confusing": "unclear",
        "really tried best": "made their best effort",
        "less communication": "reduced communication",
        "bit delay": "slight delay",
        "appreciated": "grateful"
    }
    new_text = text
    for word, repl in synonyms.items():
        new_text = re.sub(word, repl, new_text, flags=re.IGNORECASE)
    return new_text


# main: Run pipelines and print outputs
def main():
    texts = [text1, text2]
    pipelines = [
        ("Pipeline 1 (Rule-based Automaton)", pipeline1_rule_based),
        ("Pipeline 2 (TF-IDF similarity)", pipeline2_tfidf),
        ("Pipeline 3 (Local Synonym Reformation)", pipeline3_local)
    ]

    for idx, txt in enumerate(texts, start=1):
        print(f"\n=== Original Text {idx} ===\n{txt[:200]}...\n")  # truncated
        for name, func in pipelines:
            output = func(txt)
            print(f"{name} Output:\n{output[:400]}...\n")  # truncated

if __name__ == "__main__":
    main()