import re
import random


class Preprocess_data:
    def __init__(self, pathPos, pathNeg):
        self.posDict = {}
        self.negDict = {}
        self.posDictTwoWords = {}
        self.negDictTwoWords = {}
        self.trainSentencesPos = []
        self.testSentencesPos = []
        self.trainSentencesNeg = []
        self.testSentencesNeg = []
        self.read_files(pathPos, pathNeg)
        self.split_train_test()
        self.make_dict()
        # self.clean_dict()

    """
    This function get the path of both positive and negative files.
    Read both files and store them in an array.
    """
    def read_files(self, pathPos, pathNeg):
        with open(pathPos, encoding="utf8") as reader:
            for line in reader:
                line = line.replace("--", " ")
                line = re.sub('[!,*)@#%(&$_/?.^";:]', '', line)
                line = re.sub(r"[\([{})\]]", "", line)
                self.trainSentencesPos.append(line)
        with open(pathNeg, encoding="utf8") as reader:
            for line in reader:
                line = line.replace("--", " ")
                line = re.sub('[!,*)@#%(&$_/?.^";:]', '', line)
                line = re.sub(r"[\([{})\]]", "", line)
                self.trainSentencesNeg.append(line)

    """
        Split the train and test from the given sentences.
        0.1 * sentences for training and the rest for testing the machine. 
    """
    def split_train_test(self):
        num_test_pos = int(len(self.trainSentencesPos) * 0.01)
        num_test_neg = int(len(self.trainSentencesNeg) * 0.01)
        random.shuffle(self.trainSentencesPos)
        random.shuffle(self.trainSentencesNeg)
        self.testSentencesPos = self.trainSentencesPos[0:num_test_pos]
        self.testSentencesNeg = self.trainSentencesNeg[0:num_test_neg]
        del self.trainSentencesPos[0:num_test_pos]
        del self.trainSentencesNeg[0:num_test_neg]

    """
        This function makes a dictionary for positive and negative comments.
    """
    def make_dict(self):
        self.posDict['<s>'] = len(self.trainSentencesPos)
        self.posDict['</s>'] = len(self.trainSentencesPos)
        for line in self.trainSentencesPos:
            line = line.split()  # split by space
            # print(line)
            for i, word in enumerate(line):
                word = word.strip("'")
                if word in self.posDict:
                    self.posDict[word] = self.posDict.get(word) + 1
                else:
                    self.posDict[word] = 1

                if i == len(line) - 1:
                    twoWords = word + ' </s>'
                else:
                    twoWords = word + ' ' + line[i + 1].strip("'")

                if twoWords in self.posDictTwoWords:
                    self.posDictTwoWords[twoWords] = self.posDictTwoWords.get(twoWords) + 1
                else:
                    self.posDictTwoWords[twoWords] = 1
            twoWords = '<s> ' + line[0].strip("'")
            if twoWords in self.posDictTwoWords:
                self.posDictTwoWords[twoWords] = self.posDictTwoWords.get(twoWords) + 1
            else:
                self.posDictTwoWords[twoWords] = 1

        self.negDict['<s>'] = len(self.trainSentencesNeg)
        self.negDict['</s>'] = len(self.trainSentencesNeg)
        for line in self.trainSentencesNeg:
            line = line.split()
            # print(line)
            for i, word in enumerate(line):
                word = word.strip("'")
                if word in self.negDict:
                    self.negDict[word] = self.negDict.get(word) + 1
                else:
                    self.negDict[word] = 1

                if i == len(line) - 1:
                    twoWords = word + ' </s>'
                else:
                    twoWords = word + ' ' + line[i + 1].strip("'")

                if twoWords in self.negDictTwoWords:
                    self.negDictTwoWords[twoWords] = self.negDictTwoWords.get(twoWords) + 1
                else:
                    self.negDictTwoWords[twoWords] = 1
            twoWords = '<s> ' + line[0].strip("'")
            if twoWords in self.negDictTwoWords:
                self.negDictTwoWords[twoWords] = self.negDictTwoWords.get(twoWords) + 1
            else:
                self.negDictTwoWords[twoWords] = 1

    """
       This function delete repetitive words and words that was seen less than two times! 
    """
    def clean_dict(self):
        for key in list(self.posDict.keys()):
            if self.posDict[key] < 2:
                del self.posDict[key]
                if key in self.negDict:
                    del self.negDict[key]

        for key in list(self.negDict.keys()):
            if self.negDict[key] < 2:
                del self.negDict[key]
                if key in self.posDict:
                    del self.posDict[key]

        self.posDict = dict(sorted(self.posDict.items(), key=lambda item: item[1]))
        for i in range(10):
            item = self.posDict.popitem()
            if item in self.negDict:
                del self.negDict[item[0]]
        self.negDict = dict(sorted(self.negDict.items(), key=lambda item: item[1]))
        for i in range(10):
            item = self.negDict.popitem()
            if item in self.posDict:
                del self.posDict[item[0]]


class SentimentAlgorithm:
    def __init__(self, data, bigram, clean):
        if clean:
            data.clean_dict()
        self.posDict = data.posDict
        self.negDict = data.negDict
        self.posDictTwoWords = data.posDictTwoWords
        self.negDictTwoWords = data.negDictTwoWords
        self.posPwi = {}
        self.negPwi = {}
        self.posPpairwords = {}
        self.negPpairwords = {}
        self.trainSentencesPos = data.trainSentencesPos
        self.testSentencesPos = data.testSentencesPos
        self.trainSentencesNeg = data.trainSentencesNeg
        self.testSentencesNeg = data.testSentencesNeg
        self.lambda1 = None
        self.lambda2 = None
        self.lambda3 = None
        self.epsilon = None
        self.sumValuesPos = sum(self.posDict.values())
        self.sumValuesNeg = sum(self.negDict.values())
        self.bigram = bigram

    """
       P(wi) = count(wi)/M 
    """
    def calc_p_wi(self):
        for key, value in self.posDict.items():
            self.posPwi[key] = value/self.sumValuesPos

        for key, value in self.negDict.items():
            self.negPwi[key] = value/self.sumValuesNeg

    """
       P(wi|wi-1) = count(wi-1 wi)/count(wi-1)
    """
    def calc_p_wordpair(self):
        for key, value in self.posDictTwoWords.items():
            w1 = key.split()[0]
            if w1 in self.posDict:
                # print(key+"   "+w1)
                self.posPpairwords[key] = value/self.posDict[w1]
        for key, value in self.negDictTwoWords.items():
            w1 = key.split()[0]
            if w1 in self.negDict:
                # print(key+"   "+w1)
                self.negPpairwords[key] = value/self.negDict[w1]

    """
       Call the other functions to calculate probabilities of both bigram and unigram  
    """
    def train(self):
        self.calc_p_wi()
        if self.bigram:
            self.calc_p_wordpair()

    """   
       𝑃(𝑤1𝑤2…𝑤𝑛−1𝑤𝑛)=𝑃(𝑤1)∗ Π𝑃(𝑤𝑖|𝑤𝑖−1)
       𝑃(𝑤𝑖|𝑤𝑖−1)= 𝜆3𝑃(𝑤𝑖|𝑤𝑖−1)+ 𝜆2𝑃(𝑤𝑖)+𝜆1𝜖 
       𝜆3+𝜆2+𝜆1=1 
       0<𝜖<1
    """
    def calc_backoff(self, w1, w2, pos):
        sum = 0
        ww = w1 + ' ' + w2
        if pos:
            if self.bigram:
                if ww in self.posPpairwords and w2 in self.posPwi:
                    sum += (self.lambda3*self.posPpairwords.get(ww))
                    # print("ww: ",sum)
            if w2 in self.posPwi:
                sum += (self.lambda2*self.posPwi.get(w2))
                # print("w:", w2,"wi: ", sum)
        else:
            if self.bigram:
                if ww in self.negPpairwords and w2 in self.negPwi:
                    sum += (self.lambda3*self.negPpairwords.get(ww))
                    # print("ww: ", ww, "sum:", sum)
            if w2 in self.negPwi:
                sum += (self.lambda2*self.negPwi.get(w2))
                # print("w:", w2, "wi: ", sum)
        sum += (self.lambda1*self.epsilon)
        # print("epsilon: ", sum)
        return sum

    """
       Call the calc_backoff and calculate probability for each language!
       If the positive probability is more than negative for a sentence then 
       it returns true else returns false.
    """
    def check_sentence(self, sentence):
        sentence = sentence.split()
        probability_pos = 0.5
        probability_pos *= self.calc_backoff('<s>', sentence[0], True)
        for i in range(len(sentence)):
            if i == len(sentence) - 1:
                if self.bigram:
                    probability_pos *= self.calc_backoff(sentence[i], '</s>', True)
            else:
                probability_pos *= self.calc_backoff(sentence[i], sentence[i + 1], True)
        probability_neg = 0.5
        probability_neg *= self.calc_backoff('<s>', sentence[0], False)
        for i in range(len(sentence)):
            if i == len(sentence) - 1:
                if self.bigram:
                    probability_neg *= self.calc_backoff(sentence[i], '</s>', False)
            else:
                probability_neg *= self.calc_backoff(sentence[i], sentence[i + 1], False)

        # print(probability_pos)
        # print(probability_neg)
        if probability_pos > probability_neg:
            return True
        return False

    def set_parameters(self, lambda1, lambda2, lambda3, epsilon):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.epsilon = epsilon

    def set_parameters_unigram(self, lambda1, lambda2, epsilon):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.epsilon = epsilon
    """
       After training the machine, this function will test and
       calculate the precision.
    """
    def test_acc(self):
        numCorrect = 0
        allNeg = len(self.testSentencesNeg)
        for negSentence in self.testSentencesNeg:
            result = self.check_sentence(negSentence)
            if not result:
                numCorrect += 1

        allPos = len(self.testSentencesPos)
        for posSentence in self.testSentencesPos:
            result = self.check_sentence(posSentence)
            if result:
                numCorrect += 1

        precision = (numCorrect/(allPos+allNeg))*100

        print("precision: ", precision)


if __name__ == '__main__':
    data = Preprocess_data('rt-polarity.pos', 'rt-polarity.neg')

    print("BIGRAM")
    aiAgent = SentimentAlgorithm(data, True, False)
    aiAgent.train()
    aiAgent.set_parameters(0.005, 0.100, 0.895, 0.00001)
    aiAgent.test_acc()
    print("Clean")
    aiAgent2 = SentimentAlgorithm(data, True, True)
    aiAgent2.train()
    aiAgent2.set_parameters(0.005, 0.100, 0.895, 0.00001)
    aiAgent2.test_acc()

    print("UNIGRAM:")
    aiAgent_unigram = SentimentAlgorithm(data, False, False)
    aiAgent_unigram.train()
    aiAgent_unigram.set_parameters_unigram(0.1, 0.9, 0.1)
    aiAgent_unigram.test_acc()
    print("Clean")
    aiAgent_unigram2 = SentimentAlgorithm(data, False, True)
    aiAgent_unigram2.train()
    aiAgent_unigram2.set_parameters_unigram(0.1, 0.9, 0.1)
    aiAgent_unigram2.test_acc()

    # while True:
    #     comment = input()
    #     if comment == '!q':
    #         break
    #     if aiAgent.check_sentence(comment):
    #         print("NOT FILTER THIS")
    #     else:
    #         print("FILTER THIS")





