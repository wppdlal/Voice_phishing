# aug based SR
import random
import pickle
import re
import pandas as pd


# 한글 사전 불러오기
wordnet = {}
with open('./helper_function/wordnet.pickle', 'rb') as f:
	wordnet = pickle.load(f)

# 텍스트 한글만 남기고 삭제
def get_only_hangul(line):
	parseText= re.compile('/ ^[ㄱ-ㅎㅏ-ㅣ가-힣]*$/').sub('', line)
	return parseText

# 한글 사전에서 text 랜덤으로 대체
def synonym_replacement(words, n):

	new_words = words.copy()
	random_word_list = list(set([word for word in words]))
	random.shuffle(random_word_list)
	num_replaced = 0

	for random_word in random_word_list:
		synonyms = get_synonyms(random_word)

		if len(synonyms) > 1:
			synonym = random.choice(list(synonyms))
			new_words = [synonym if word == random_word else word for word in new_words]
			num_replaced += 1

		if num_replaced >= n:
			break

	if len(new_words) != 0:
		sentence = ' '.join(new_words)
		new_words = sentence.split(' ')

	else:
		new_words = ''

	return new_words


# 유의어 가져오기
def get_synonyms(word):
	synomyms = []

	try:
		for syn in wordnet[word]:
			for s in syn:
				synomyms.append(s)
	except:
		pass

	return synomyms

# SR aug
def SR(sentence, alpha_sr=0.15, num_aug=9):

	sentence = get_only_hangul(sentence)
	words = sentence.split(' ')
	words = [word for word in words if word is not '']
	num_words = len(words)

	augmented_sentences = []
	num_new_per_technique = int(num_aug/4) + 1

	n_sr = max(1, int(alpha_sr*num_words))

	# sr
	for _ in range(num_new_per_technique):
		a_words = synonym_replacement(words, n_sr)
		augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [get_only_hangul(sentence) for sentence in augmented_sentences]
	random.shuffle(augmented_sentences)

	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug / len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	augmented_sentences.append(sentence)

	return augmented_sentences[1]

# sample

#r = SR('동해물과 백두산이 마르고 닳도록 하느님이 보우하사 우리나라 만세 무궁화 삼천리 화려 강산 대한 사람 대한으로 길이 보전하세')
#print(r)