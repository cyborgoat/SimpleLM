PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

words = [
    "hello", "world", "i", "am", "a", "robot", "how", "are", "you", "today",
    "good", "great", "awesome", "happy", "learning", "is", "fun", "tell", "me",
    "story", "what", "time", "thank", "bye", "ai", "can", "help", "your", "name",
    "joke", "computer", "doctor", "virus", "stands", "for", "artificial", "intelligence",
    "weather", "connected", "internet", "hope", "sunny", "welcome", "goodbye", "see",
    "next", "answer", "questions", "chat", "who", "created", "developer", "demonstrate",
    "concepts", "favorite", "color", "all", "colors", "equally", "of", "course", "need",
    "with", "capital", "france", "paris", "plus", "equals", "do", "like", "music",
    "enjoy", "about", "sing", "write", "lyrics", "machine", "field", "focused", "from",
    "data", "old", "age", "program", "fun", "fact", "did", "know", "honey", "never",
    "spoils", "try", "translate", "simple", "phrases", "purpose", "assist", "answer",
    "work", "process", "input", "generate", "responses", "using", "neural", "network",
    "languages", "speak", "english", "best", "code", "examples", "meaning", "life",
    "philosophical", "question", "president", "united", "states", "updated", "real",
    "time", "check", "latest", "news", "food", "eat", "read", "pizza", "lot", "poem",
    "roses", "red", "violets", "blue", "here", "chat", "how", "learn", "python", "start",
    "basics", "practice", "coding", "documentation"
]
VOCAB_SIZE = max(70, 3 + len(words))
vocab = {word: i + 3 for i, word in enumerate(words)}
vocab["<pad>"] = PAD_TOKEN
vocab["<sos>"] = SOS_TOKEN
vocab["<eos>"] = EOS_TOKEN
idx_to_vocab = {i: word for word, i in vocab.items()}

def tokenize(text, add_sos_eos=True):
    tokens = [vocab.get(word, vocab["<pad>"]) for word in text.lower().split()]
    if add_sos_eos:
        return [SOS_TOKEN] + tokens + [EOS_TOKEN]
    return tokens

def detokenize(token_ids):
    words = []
    for token_id in token_ids:
        if token_id == EOS_TOKEN:
            break
        if token_id not in [PAD_TOKEN, SOS_TOKEN]:
            words.append(idx_to_vocab.get(token_id, "<unk>"))
    return " ".join(words)
