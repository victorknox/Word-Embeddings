# imports
from numpy import array
from numpy import diag
from numpy import zeros
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD
import re

def tokenize(text):
    
    text = str(text)
    # remove non-ascii characters
    cleaned_text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # make the text lowercase
    cleaned_text = cleaned_text.lower()

    # replace urls
    cleaned_text = re.sub(r'http\S+', '<URL>', cleaned_text)

    # replace hashtags
    cleaned_text = re.sub(r'#\w+', '<HASHTAG>', cleaned_text)

    # replace mentions
    cleaned_text = re.sub(r'@\w+', '<MENTION>', cleaned_text)

    # replace percentages
    cleaned_text = re.sub(r'\d+%', '<PERCENTAGE>', cleaned_text)

    # replace all date formats
    cleaned_text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '<DATE>', cleaned_text)

    # replace all time formats
    cleaned_text = re.sub(r'\d{1,2}[:]\d{2}([:]\d{2})?(am|pm)?', '<TIME>', cleaned_text)

    # replace any repeated punctuation to a single one
    cleaned_text = re.sub(r'([.,!?:;><])\1+', r'\1', cleaned_text)

    # replace any letter repeated more than twice to a single one (eg: tiired to tired. oopppps -> oops)
    cleaned_text = re.sub(r'([a-zA-Z])\1{2,}', r'\1', cleaned_text)

    # replace can't with cannot
    cleaned_text = re.sub(r'can\'t', 'cannot', cleaned_text)

    # replace xn't with x + not
    cleaned_text = re.sub(r'n\'t', r' not', cleaned_text)

    # replace x'm with x + am
    cleaned_text = re.sub(r'\'m', r' am', cleaned_text)

    # replace x's with x + is
    cleaned_text = re.sub(r'\'s', r' is', cleaned_text)

    # replace x're with x + are
    cleaned_text = re.sub(r'\'re', r' are', cleaned_text)

    # replace x'll to x + will
    cleaned_text = re.sub(r'\'ll', r' will', cleaned_text)

    # replace x'd to x + would
    cleaned_text = re.sub(r'\'d', r' would', cleaned_text)

    # replace x've to x + have
    cleaned_text = re.sub(r'\'ve', r' have', cleaned_text)

    # splitting words along with punctuation before or after (eg: good. -> good + . , "lmao" +> " + lmao + ", [Dang] -> [ + Dang + ])
    cleaned_text = re.sub(r'(\w+)([.,!?:;\[\]*/"\'\(\)])', r'\1 \2', cleaned_text)
    cleaned_text = re.sub(r'([.,!?:;\[\]*/"\'\(\)])(\w+)', r'\1 \2', cleaned_text)

    # remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # split words with hyphens
    cleaned_text = re.sub(r'(\w+)-(\w+)', r'\1 \2', cleaned_text)

    # tag numbers
    cleaned_text = re.sub(r'\d+', '<NUMBER>', cleaned_text)
    

    return cleaned_text



def make_svd(A, k):
	# computing the SVD of A
	svd = TruncatedSVD(n_components=k)
	svd.fit(A)
	T = svd.transform(A)	
	return T


def make_vocab(data):
	# creating dictionary of words
	vocab = {}
	vocab_list = []
	for sentence in data:
		words = sentence.split()
		for word in words:
			if word not in vocab:
				vocab_list += [word]
				vocab[word] = 0
			vocab[word] += 1

	# handling unknowns 
	vocab['unk'] = 0
	vocab_list += ['unk']

	# ignore words with frequency less than 5
	for word in vocab_list:
		if vocab[word] < 5:
			vocab_list.remove(word)
			vocab['unk'] += vocab[word]
			del vocab[word]
			

	return vocab, vocab_list 


def make_cmatrix(data, vocab, vocab_list, window_size):
	# creating co-occurrence matrix
	n = len(vocab_list)
	x = int((window_size - 1 )/2)
	cmatrix = zeros((n, n))
	for sentence in data:
		words = sentence.split()
		for i in range(len(words)):
			if(vocab.get(words[i], 0) == 0):
				words[i] = 'unk'
		for i in range(len(words)):				
			for j in range(1, x + 1):
				if(i + j < len(words)):
					cmatrix[vocab_list.index(words[i]), vocab_list.index(words[i + j])] += 1
				if(i - j >= 0):
					cmatrix[vocab_list.index(words[i]), vocab_list.index(words[i - j])] += 1

	return cmatrix
