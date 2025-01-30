# aug based on RI
import random
import pickle
import re


# 동의어 사전
wordnet = {}
with open("./helper_function/wordnet.pickle", "rb") as f:
	wordnet = pickle.load(f)

def get_synonyms(word):
	synomyms = []

	try:
		for syn in wordnet[word]:
			for s in syn:
				synomyms.append(s)
	except:
		pass

	return synomyms

def random_insertion(words, n):
	new_words = words.copy()

	for _ in range(n):
		add_word(new_words)
	
	return new_words

# 랜덤으로 추가할 단어
def add_word(new_words):
	synonyms = []
	counter = 0

	while len(synonyms) < 1:

		if len(new_words) >= 1:
			random_word = new_words[random.randint(0, len(new_words) - 1)]
			synonyms = get_synonyms(random_word)
			counter += 1
			
		else:
			random_word = ''

		if counter >= 10:
			return
		
	random_synonym = synonyms[0]
	random_idx = random.randint(0, len(new_words) - 1)
	new_words.insert(random_idx, random_synonym)

# RI aug
def RI(sentence, alpha_ri=0.15, num_aug=9):

	# sentence = get_only_hangul(sentence)
	words = sentence.split(' ')
	words = [word for word in words if word is not '']
	num_words = len(words)

	augmented_sentences = []
	num_new_per_technique = int(num_aug / 4) + 1

	n_ri = max(1, int(alpha_ri * num_words))

	# ri
	for _ in range(num_new_per_technique):
		a_words = random_insertion(words, n_ri)
		augmented_sentences.append(' '.join(a_words))


	# augmented_sentences = [get_only_hangul(sentence) for sentence in augmented_sentences]
	random.shuffle(augmented_sentences)

	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	augmented_sentences.append(sentence)

	return augmented_sentences[1]

# sample

#r = RI('동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세 무궁화 삼천리 화려 강산 대한 사람 대한으로 길이 보전하세')
#print(r)