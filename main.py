import re
import random


class SentimentAlgorithm:
    def __init__(self, pathPos, pathNeg):
        self.posDict = {}
        self.negDict = {}
        self.posDictTwoWords = {}
        self.negDictTwoWords = {}
        self.posPwi = {}
        self.negPwi = {}
        self.posPpairwords = {}
        self.negPpairwords = {}
        self.trainSentencesPos = []
        self.testSentencesPos = []
        self.trainSentencesNeg = []
        self.testSentencesNeg = []
        self.read_files(pathPos, pathNeg)
        self.split_train_test()
        self.make_dict(True)
        # self.clean_dict()
        self.sumValuesPos = sum(self.posDict.values())
        self.sumValuesNeg = sum(self.negDict.values())

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

    def split_train_test(self):
        num_test_pos = int(len(self.trainSentencesPos) * 0.1)
        num_test_neg = int(len(self.trainSentencesNeg) * 0.1)
        random.shuffle(self.trainSentencesPos)
        random.shuffle(self.trainSentencesNeg)
        self.testSentencesPos = self.trainSentencesPos[0:num_test_pos]
        self.testSentencesNeg = self.trainSentencesNeg[0:num_test_neg]
        del self.trainSentencesPos[0:num_test_pos]
        del self.trainSentencesNeg[0:num_test_neg]

    def make_dict(self, bigram):
        self.posDict['<s>'] = len(self.trainSentencesPos)
        self.posDict['</s>'] = len(self.trainSentencesPos)
        for line in self.trainSentencesPos:
            line = line.split()
            # print(line)
            for i, word in enumerate(line):
                word = word.strip("'")
                if word in self.posDict:
                    self.posDict[word] = self.posDict.get(word) + 1
                else:
                    self.posDict[word] = 1
                if bigram:
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
                if bigram:
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

    def clean_dict(self):
        self.posDict = {key: val for key, val in self.posDict.items() if val > 2}
        self.negDict = {key: val for key, val in self.negDict.items() if val > 2}
        self.posDict = dict(sorted(self.posDict.items(), key=lambda item: item[1]))
        for i in range(10):
            self.posDict.popitem()
        self.negDict = dict(sorted(self.negDict.items(), key=lambda item: item[1]))
        for i in range(10):
            self.negDict.popitem()

    def calc_pwi(self):
        for key, value in self.posDict.items():
            self.posPwi[key] = value/self.sumValuesPos

        for key, value in self.negDict.items():
            self.negPwi[key] = value/self.sumValuesNeg

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

    def train(self):
        self.calc_pwi()
        self.calc_p_wordpair()

    def check_comment(self):
        return True


if __name__ == '__main__':
    aiAgent = SentimentAlgorithm('rt-polarity.pos', 'rt-polarity.neg')
    aiAgent.train()
    print(aiAgent.negPpairwords)
    # print(aiAgent.posDictTwoWords)
    # print(aiAgent.posDict)
    # print(aiAgent.negDict)

    # print(aiAgent.sumValuesPos)
    # print(aiAgent.sumValuesNeg)
    # print(aiAgent.posPwi)
    # print(aiAgent.negPwi)

    # print(aiAgent.trainSentencesPos)
    # print(len(aiAgent.trainSentencesPos))
    # # print(aiAgent.testSentencesPos)
    # print(len(aiAgent.testSentencesPos))
    # while True:
    #     comment = input()
    #     if comment == '!q':
    #         break
    #     # TODO: IMPLEMENT ALGORITHM
