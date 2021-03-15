import nltk
import sys

#estrae il numero di frasi e di token, la lista di token e il testo taggato
def inizializza(file):
    fhand = open(file, 'r', encoding = "utf8")
    text = fhand.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi = sent_tokenizer.tokenize(text)
    tokensTOT = []
    for frase in frasi:
        token = nltk.word_tokenize(frase)
        tokensTOT+=token
    numFrasi = len(frasi)
    numToken = len(tokensTOT)
    POStag = nltk.pos_tag(tokensTOT)

    return numFrasi, numToken, tokensTOT, POStag



#calcola lunghezza media di frasi e token
def lungMediaFrasiEToken(numFrasi, tokensTOT):
    lungMediaFrasi = len(tokensTOT)/numFrasi
    lungTotToken = 0
    #calcola la lunghezza totale dei caratteri nei token,
    for token in tokensTOT:
        lungTotToken += len(token)
    lungMediaToken = lungTotToken/len(tokensTOT)

    return lungMediaFrasi, lungMediaToken



"""calcolate la grandezza del vocabolario e la ricchezza lessicale calcolata attraverso la Type Token Ratio
(TTR), in entrambi i casi calcolati nei primi 5000 token;"""
def lungVocabolarioETTR_5000(tokensTOT):
    #restituisce la lunghezza del vocabolario e la TTR per i primi 5000 token
    vocabolario5000 = list(set(tokensTOT[:5000]))
    lungVocabolario = len(vocabolario5000)
    TTR = lungVocabolario/5000

    return lungVocabolario, TTR



"""calcolate la distribuzione delle classi di frequenza |V1|, |V5| e |V10| all'aumentare del corpus per
porzioni incrementali di 500 token (500 token, 1000 token, 1500 token, etc.);"""
def CdF_500(tokensTOT):
    #contengono i vocaboli appartenenti a una specifica classe di frequenza
    V1_class = []
    V5_class = []
    V10_class = []
    #contengono la lunghezza delle liste di cui sopra, misurata ogni 500 token.
    #var sta per variazione
    V1_var = []
    V5_var = []
    V10_var = []

    numToken = len(tokensTOT)
    vocabolario = list(set(tokensTOT))

    #itera partendo da 500 e fino alla lunghezza del corpus con incrementi di 500
    for r in range(500, numToken, 500):
        for v in vocabolario:
            #calcola la frequenza assoluta del token fino al valore dell'iterazione
            freqToken = tokensTOT[:r].count(v)
            #se la frequenza è 1, 5 o 10, inserisce il token in una lista apposita...
            if (freqToken == 1):
                V1_class.append(v)
            elif(freqToken == 5):
                V5_class.append(v)
            elif(freqToken == 10):
                V10_class.append(v)
        #...e poi inserisce in un'altra lista il numero di parole con
        #quella frequenza fino a quell'iterazione.
        V1_var.append(len(V1_class))
        V1_class = []
        V5_var.append(len(V5_class))
        V5_class = []
        V10_var.append(len(V10_class))
        V10_class = []

    return V1_var, V5_var, V10_var



#calcola media di sostantivi e verbi per frase
def mediaSostantiviEVerbi(POStag, numFrasi):
    numSostantivi = 0
    numVerbi = 0

    #cicla il testo taggato. Se trova un elemento di interesse lo inserisce in una lista
    for p in POStag:
        if p[1] in ["NNP","NNPS","NN","NNS"]:
            numSostantivi = numSostantivi + 1
        elif p[1] in ["VB","VBD","VBG","VBN","VBP","VBZ","MD"]:
            numVerbi = numVerbi + 1

    mediaSostantivi = numSostantivi/numFrasi
    mediaVerbi = numVerbi/numFrasi

    return mediaSostantivi, mediaVerbi, numSostantivi, numVerbi



"""calcola la densità lessicale, calcolata come il rapporto tra il numero totale di occorrenze nel testo di
Sostantivi, Verbi, Avverbi, Aggettivi e il numero totale di parole nel testo (ad esclusione dei
segni di punteggiatura marcati con POS "," "."):
(|Sostantivi|+|Verbi|+|Avverbi|+|Aggettivi|)/(TOT-( |.|+|,| ) )."""
def calcolaDL(POStag, numSostantivi, numVerbi):
    numAvverbi = 0
    numAggettivi = 0
    #numero di token senza punteggiatura
    numTokenSP = 0
    #cicla il testo taggato. Se trova un elemento di interesse aumenta un contatore
    for p in POStag:
        if p[1] in ["RB","RBR","RBS","WRB"]:
            numAvverbi = numAvverbi + 1
        elif p[1] in ["JJ","JJR","JJS"]:
            numAggettivi = numAggettivi + 1
        if p[1] not in [".", ","]:
            numTokenSP = numTokenSP + 1
    densitàLessicale = (numSostantivi + numVerbi + numAvverbi + numAggettivi) / numTokenSP
    return densitàLessicale


#il main confronta due testi invocando le funzioni di cui sopra e ne stampa i risultati
def main(file1, file2):
    numFrasi1, numToken1, tokensTOT1, POStag1 = inizializza(file1)
    numFrasi2, numToken2, tokensTOT2, POStag2 = inizializza(file2)
    print("Il primo testo conta", numFrasi1, "frasi, per un totale di", numToken1, "token.")
    print("Il secondo testo conta", numFrasi2, "frasi, per un totale di", numToken2, "token.")
    print()

    lungMediaFrasi1, lungMediaToken1 = lungMediaFrasiEToken(numFrasi1, tokensTOT1)
    lungMediaFrasi2, lungMediaToken2 = lungMediaFrasiEToken(numFrasi2, tokensTOT2)
    print("Nel primo testo, la lunghezza media delle frasi è pari a", lungMediaFrasi1,
    "token, mentre la lunghezza media dei token è pari a", lungMediaToken1, "caratteri.")
    print("Nel secondo testo, la lunghezza media delle frasi è pari a", lungMediaFrasi2,
    "token, mentre la lunghezza media dei token è pari a", lungMediaToken2, "caratteri.")
    print()

    lungVocabolario1_5000, TTR1 = lungVocabolarioETTR_5000(tokensTOT1)
    lungVocabolario2_5000, TTR2 = lungVocabolarioETTR_5000(tokensTOT2)
    print("Contando solo i primi 5000 token, il vocabolario del primo testo è lungo", lungVocabolario1_5000,
      "parole tipo, mentre la type-token ratio è pari a", TTR1, ".")
    print("Contando solo i primi 5000 token, il vocabolario del secondo testo è lungo", lungVocabolario2_5000,
      "parole tipo, mentre la type-token ratio è pari a", TTR2, ".")
    print()

    V1_var1, V5_var1, V10_var1 = CdF_500(tokensTOT1)
    V1_var2, V5_var2, V10_var2 = CdF_500(tokensTOT2)
    print("Segue l'andamento della classe di frequenza V1 del primo testo con incrementi di 500 token per volta:")
    for i in V1_var1:
        print(i)
    print("Segue l'andamento della classe di frequenza V1 del secondo testo con incrementi di 500 token per volta:")
    for i in V1_var2:
        print(i)
    print()

    print("Segue l'andamento della classe di frequenza V5 del primo testo con incrementi di 500 token per volta:")
    for i in V5_var1:
        print(i)
    print("Segue l'andamento della classe di frequenza V5 del secondo testo con incrementi di 500 token per volta:")
    for i in V5_var2:
        print(i)
    print()

    print("Segue l'andamento della classe di frequenza V10 del primo testo con incrementi di 500 token per volta:")
    for i in V10_var1:
        print(i)
    print("Segue l'andamento della classe di frequenza V10 del secondo testo con incrementi di 500 token per volta:")
    for i in V10_var2:
        print(i)
    print()

    mediaSostantivi1, mediaVerbi1, numSostantivi1, numVerbi1 = mediaSostantiviEVerbi(POStag1, numFrasi1)
    mediaSostantivi2, mediaVerbi2, numSostantivi2, numVerbi2 = mediaSostantiviEVerbi(POStag2, numFrasi2)
    print("Nel primo testo, il numero medio di sostantivi per frase è pari a", mediaSostantivi1, ", nel secondo è", mediaSostantivi2)
    print("Nel primo testo, il numero medio di verbi per frase è pari a", mediaVerbi1, ", nel secondo è", mediaVerbi2)
    print()

    densitàLessicale1 = calcolaDL(POStag1, numSostantivi1, numVerbi1)
    densitàLessicale2 = calcolaDL(POStag2, numSostantivi2, numVerbi2)
    print("La densità lessicale del primo testo è pari a", densitàLessicale1, ", quella del secondo è", densitàLessicale2)

main(sys.argv[1], sys.argv[2])
