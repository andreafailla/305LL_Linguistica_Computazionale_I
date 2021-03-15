import nltk
import sys
import math

    #estrae la lista di token, il testo taggato e la lista di frasi
def inizializza(file):
    fhand = open(file, 'r', encoding = "utf8")
    text = fhand.read()
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    frasi = sent_tokenizer.tokenize(text)
    tokensTOT = []
    for frase in frasi:
        token = nltk.word_tokenize(frase)
        tokensTOT+=token
    POStag = nltk.pos_tag(tokensTOT)

    return tokensTOT, POStag, frasi



""" estraete ed ordinate in ordine di frequenza decrescente, indicando anche la relativa
frequenza: le 10 PoS (Part-of-Speech) più frequenti;
◦ i 20 sostantivi e i 20 verbi più frequenti;
◦ i 20 bigrammi composti da un Sostantivo seguito da un Verbo più frequenti;
◦ i 20 bigrammi composti da un Aggettivo seguito da un Sostantivo più frequenti; """
def piuFrequenti(POStag):
    POS = []
    sostantivi = []
    verbi = []
    bigrammiSV = []
    bigrammiAS = []

    for p in POStag:
        #inserisce i tag in una lista apposita. Serve a calcolare i tag più frequenti
        POS.append(p[1])
        #se trova il tag di un sostantivo o verbo, inserisce il relativo token in una lista apposita
        if p[1] in ["NNP","NNPS","NN","NNS"]:
            sostantivi.append(p[0])
        elif p[1] in ["VB","VBD","VBG","VBN","VBP","VBG","VBZ","MD"]:
            verbi.append(p[0])

    bigrammi = list(nltk.bigrams(POStag))
    for b in bigrammi:
        #come sopra, ma coi bigrammi
        if ((b[0][1] in ["NNP","NNPS","NN","NNS"]) and (b[1][1] in ["VB","VBD","VBG","VBN","VBP","VBG","VBZ","MD"])):
            bigrammiSV.append((b[0][0], b[1][0]))
        if ((b[0][1] in ["JJ","JJR","JJS"]) and (b[1][1] in ["NNP","NNPS","NN","NNS"])):
            bigrammiAS.append((b[0][0], b[1][0]))

    #calcola i più frequenti
    POSPiuFrequenti = nltk.FreqDist(POS).most_common(10)
    sostantiviPiuFrequenti = nltk.FreqDist(sostantivi).most_common(20)
    verbiPiuFrequenti = nltk.FreqDist(verbi).most_common(20)
    bigrSVPiuFrequenti = nltk.FreqDist(bigrammiSV).most_common(20)
    bigrASPiuFrequenti = nltk.FreqDist(bigrammiAS).most_common(20)

    return POSPiuFrequenti, sostantiviPiuFrequenti, verbiPiuFrequenti, \
    bigrSVPiuFrequenti, bigrASPiuFrequenti



"""estraete ed ordinate i 20 bigrammi di token (dove ogni token deve avere una frequenza
maggiore di 3):
◦ con probabilità congiunta massima, indicando anche la relativa probabilità;
◦ con probabilità condizionata massima, indicando anche la relativa probabilità;
◦ con forza associativa (calcolata in termini di Local Mutual Information) massima,
indicando anche la relativa forza associativa;"""
def bigrammiMAX(tokensTOT):
    bigrammi = list(nltk.bigrams(tokensTOT))

    #contiene i bigrammi i cui token hanno entrambi frequenza maggiore di tre
    bigrFreqMagg3 = []
    listaProbCond = []
    listaProbCong = []
    listaLMI = []

    for b in bigrammi:
        if ((tokensTOT.count(b[0]) > 3) and (tokensTOT.count(b[1]) > 3)):
            bigrFreqMagg3.append(b)

    for b in list(set(bigrFreqMagg3)):
        #calcola probabilità e LMI, poi inserisce la tupla (bigramma, valore)
        #nell'apposita lista
        probCond = (bigrammi.count(b))/(tokensTOT.count(b[0]))
        listaProbCond.append(((b),(probCond)))
        probCong = bigrammi.count(b)/len(tokensTOT)
        listaProbCong.append(((b),(probCong)))
        #MI = log2( P(a, b) / (p(a) * p(b)) )
        MI = math.log2((bigrammi.count(b) * len(tokensTOT)) / (tokensTOT.count(b[0]) * tokensTOT.count(b[1])))
        LMI = MI * bigrammi.count(b)
        listaLMI.append(((b),(LMI)))

    #ordina la lista di tuple in base al secondo elemento
    listaProbCond_sorted = sorted(listaProbCond, reverse = True, key = lambda x: x[1])
    listaProbCong_sorted = sorted(listaProbCong, reverse = True, key = lambda x: x[1])
    listaLMI_sorted = sorted(listaLMI, reverse = True, key = lambda x: x[1])
    #estrae le prime venti tuple
    listaProbCond_sorted_20 = listaProbCond_sorted[:20]
    listaProbCong_sorted_20 = listaProbCong_sorted[:20]
    listaLMI_sorted_20 = listaLMI_sorted[:20]

    return listaProbCond_sorted_20, listaProbCong_sorted_20, listaLMI_sorted_20



"""per ogni lunghezza di frase da 8 a 15 token, estraete la frase con probabilità più alta, dove la
probabilità deve essere calcolata attraverso un modello di Markov di ordine 1 usando lo
Add-one Smoothing. Il modello deve usare le statistiche estratte dal corpus che contiene le
frasi;"""
def markov(frasi, tokensTOT):
    prob8 = 0
    prob9 = 0
    prob10 = 0
    prob11 = 0
    prob12 = 0
    prob13 = 0
    prob14 = 0
    prob15 = 0
    frase8 = ""
    frase9 = ""
    frase10 = ""
    frase11 = ""
    frase12 = ""
    frase13 = ""
    frase14 = ""
    frase15 = ""
    numToken = len(tokensTOT)
    numVocab = len(set(tokensTOT))
    bigrammi = list(nltk.bigrams(tokensTOT))

    #sostituisce ogni frase con la frase tokenizzata
    for frase in frasi:
        frase = nltk.word_tokenize(frase)

        #se la lunghezza è compresa tra 8 e 16 (escluso)...
        if len(frase) in range(8, 16):
            #...calcola la probabilità del primo elemento e di tutti gli altri...
            for i in range(len(frase) - 1):
                if i == 0:
                    prob = ((tokensTOT.count(frase[i]) + 1) / (numToken + numVocab))
                else:
                    prob_2nd_fatt = ((bigrammi.count((frase[i], frase[i+1])) + 1) /\
                    (tokensTOT.count(frase[i]) + numVocab))
                    prob = prob * prob_2nd_fatt

            #... e poi controlla lunghezza e probabilità e le sovrascrive
            if len(frase) == 8 and prob > prob8:
                prob8 = prob
                frase = " ".join(frase)
                frase8 = frase
            elif len(frase) == 9 and prob > prob9:
                prob9 = prob
                frase = " ".join(frase)
                frase9 = frase
            elif len(frase) == 10 and prob > prob10:
                prob10 = prob
                frase = " ".join(frase)
                frase10 = frase
            elif len(frase) == 11 and prob > prob11:
                prob11 = prob
                frase = " ".join(frase)
                frase11 = frase
            elif len(frase) == 12 and prob > prob12:
                prob12 = prob
                frase = " ".join(frase)
                frase12 = frase
            elif len(frase) == 13 and prob > prob13:
                prob13 = prob
                frase = " ".join(frase)
                frase13 = frase
            elif len(frase) == 14 and prob > prob14:
                prob14 = prob
                frase = " ".join(frase)
                frase14 = frase
            elif len(frase) == 15 and prob > prob15:
                prob15 = prob
                frase = " ".join(frase)
                frase15 = frase

    return frase8, frase9, frase10, frase11, frase12, frase13, frase14, frase15, \
    prob8, prob9, prob10, prob11, prob12, prob13, prob14, prob15



"""dopo aver individuato e classificato le Entità Nominate (NE) presenti nel testo, estraete:
◦ i 15 nomi propri di persona più frequenti (tipi), ordinati per frequenza;
◦ i 15 nomi propri di luogo più frequenti (tipi), ordinati per frequenza."""
def NER(POStag):

    NEtag = nltk.ne_chunk(POStag)
    persone = []
    luoghi = []
    for nodo in NEtag:
        NE = ""
        #se il nodo ha l'attributo "label"...
        if hasattr(nodo, "label"):
            if nodo.label() in ["PERSON", "GPE"]:
                #...se il valore dell'attributo corrisponde
                #a una persona o a un luogo, li inserisce in una lista apposita
                for pNE in nodo.leaves():
                    NE = NE + " " + pNE[0]
                if (nodo.label() == "PERSON"):
                    persone.append(NE)
                elif (nodo.label() == "GPE"):
                    luoghi.append(NE)

    personePiuFrequenti = nltk.FreqDist(persone).most_common(15)
    luoghiPiuFrequenti = nltk.FreqDist(luoghi).most_common(15)
    return personePiuFrequenti, luoghiPiuFrequenti

"""Il main confronta due testi chiamando le funzioni di cui sopra e ne stampa i risultati"""
def main(file1, file2):
    tokensTOT1, POStag1, frasi1 = inizializza(file1)
    tokensTOT2, POStag2, frasi2 = inizializza(file2)
    POSPiuFrequenti1, sostantiviPiuFrequenti1, verbiPiuFrequenti1, bigrSVPiuFrequenti1, \
    bigrASPiuFrequenti1 = piuFrequenti(POStag1)
    POSPiuFrequenti2, sostantiviPiuFrequenti2, verbiPiuFrequenti2, bigrSVPiuFrequenti2, \
    bigrASPiuFrequenti2 = piuFrequenti(POStag2)

    print("Segue una lista delle Part of Speech più frequenti nel primo testo:")
    for p in POSPiuFrequenti1:
        print("POS", p[0], "\tcon frequenza", p[1])
    print("Segue una lista delle Part of Speech più frequenti nel secondo testo:")
    for p in POSPiuFrequenti2:
        print("POS", p[0], "\tcon frequenza", p[1])
    print()

    print("Segue una lista dei sostantivi più frequenti nel primo testo:")
    for s in sostantiviPiuFrequenti1:
        print("Il sostantivo", s[0], "\tcon frequenza", s[1])
    print("Segue una lista dei sostantivi più frequenti nel secondo testo:")
    for s in sostantiviPiuFrequenti2:
        print("Il sostantivo", s[0], "\tcon frequenza", s[1])
    print()

    print("Segue una lista dei verbi più frequenti nel primo testo:")
    for v in verbiPiuFrequenti1:
        print("Il verbo", v[0], "\tcon frequenza", v[1])
    print("Segue una lista dei verbi più frequenti nel secondo testo:")
    for v in verbiPiuFrequenti2:
        print("Il verbo", v[0], "\tcon frequenza", v[1])
    print()

    print("Segue una lista dei bigrammi <sostantivo, verbo> più frequenti nel primo testo:")
    for b in bigrSVPiuFrequenti1:
        print("Il bigramma", b[0], "\tcon frequenza", b[1])
    print("Segue una lista dei bigrammi <sostantivo, verbo> più frequenti nel secondo testo:")
    for b in bigrSVPiuFrequenti2:
        print("Il bigramma", b[0], "\tcon frequenza", b[1])
    print()

    print("Segue una lista dei bigrammi <aggettivo, sostantivo> più frequenti nel primo testo:")
    for b in bigrASPiuFrequenti1:
        print("Il bigramma", b[0], "\tcon frequenza", b[1])
    print("Segue una lista dei bigrammi <aggettivo, sostantivo> più frequenti nel secondo testo:")
    for b in bigrASPiuFrequenti2:
        print("Il bigramma", b[0], "\tcon frequenza", b[1])
    print()

    listaProbCond_sorted_20_1, listaProbCong_sorted_20_1, listaLMI_sorted_20_1 = bigrammiMAX(tokensTOT1)
    listaProbCond_sorted_20_2, listaProbCong_sorted_20_2, listaLMI_sorted_20_2 = bigrammiMAX(tokensTOT2)

    print("Segue una lista dei venti bigrammi con probabilità condizionata massima nel primo testo:")
    for elem in listaProbCond_sorted_20_1:
        print("Il bigramma", elem[0], "\tcon probabilità", elem[1])
    print("Segue una lista dei venti bigrammi con probabilità condizionata massima nel secondo testo:")
    for elem in listaProbCond_sorted_20_2:
        print("Il bigramma", elem[0], "\tcon probabilità", elem[1])
    print()

    print("Segue una lista dei venti bigrammi con probabilità congiunta massima nel primo testo:")
    for elem in listaProbCong_sorted_20_1:
        print("Il bigramma", elem[0], "\tcon probabilità", elem[1])
    print("Segue una lista dei venti bigrammi con probabilità congiunta massima nel secondo testo:")
    for elem in listaProbCong_sorted_20_2:
        print("Il bigramma", elem[0], "\tcon probabilità", elem[1])
    print()

    print("Segue una lista dei venti bigrammi con forza associativa massima nel primo testo:")
    for elem in listaLMI_sorted_20_1:
        print("Il bigramma", elem[0], "\tcon LMI", elem[1])
    print("Segue una lista dei venti bigrammi con forza associativa massima nel secondo testo:")
    for elem in listaLMI_sorted_20_2:
        print("Il bigramma", elem[0], "\tcon LMI", elem[1])
    print()

    frase8_1, frase9_1, frase10_1, frase11_1, frase12_1, frase13_1, frase14_1, frase15_1, \
    prob8_1, prob9_1, prob10_1, prob11_1, prob12_1, prob13_1, prob14_1, prob15_1 = markov(frasi1, tokensTOT1)
    frase8_2, frase9_2, frase10_2, frase11_2, frase12_2, frase13_2, frase14_2, frase15_2, \
    prob8_2, prob9_2, prob10_2, prob11_2, prob12_2, prob13_2, prob14_2, prob15_2 = markov(frasi2, tokensTOT2)

    print("Nel primo testo, la frase di lunghezza 8 con probabilità massima è:\n", frase8_1, "\tcon probabilità", prob8_1)
    print("Nel secondo testo, la frase di lunghezza 8 con probabilità massima è:\n", frase8_2, "\tcon probabilità", prob8_2)
    print()

    print("Nel primo testo, la frase di lunghezza 9 con probabilità massima è:\n", frase9_1, "\tcon probabilità", prob9_1)
    print("Nel secondo testo, la frase di lunghezza 9 con probabilità massima è:\n", frase9_2, "\tcon probabilità", prob9_2)
    print()

    print("Nel primo testo, la frase di lunghezza 10 con probabilità massima è:\n", frase10_1, "\tcon probabilità", prob10_1)
    print("Nel secondo testo, la frase di lunghezza 10 con probabilità massima è:\n", frase10_2, "\tcon probabilità", prob10_2)
    print()

    print("Nel primo testo, la frase di lunghezza 11 con probabilità massima è:\n", frase11_1, "\tcon probabilità", prob11_1)
    print("Nel secondo testo, la frase di lunghezza 11 con probabilità massima è:\n", frase11_2, "\tcon probabilità", prob11_2)
    print()

    print("Nel primo testo, la frase di lunghezza 12 con probabilità massima è:\n", frase12_1, "\tcon probabilità", prob12_1)
    print("Nel secondo testo, la frase di lunghezza 12 con probabilità massima è:\n", frase12_2, "\tcon probabilità", prob12_2)
    print()

    print("Nel primo testo, la frase di lunghezza 13 con probabilità massima è:\n", frase13_1, "\tcon probabilità", prob13_1)
    print("Nel secondo testo, la frase di lunghezza 13 con probabilità massima è:\n", frase13_2, "\tcon probabilità", prob13_2)
    print()

    print("Nel primo testo, la frase di lunghezza 14 con probabilità massima è:\n", frase14_1, "\tcon probabilità", prob14_1)
    print("Nel secondo testo, la frase di lunghezza 14 con probabilità massima è:\n", frase14_2, "\tcon probabilità", prob14_2)
    print()

    print("Nel primo testo, la frase di lunghezza 15 con probabilità massima è:", frase15_1, "\tcon probabilità", prob15_1)
    print("Nel secondo testo, la frase di lunghezza 15 con probabilità massima è:\n", frase15_2, "\tcon probabilità", prob15_2)
    print()

    personePiuFrequenti1, luoghiPiuFrequenti1 = NER(POStag1)
    personePiuFrequenti2, luoghiPiuFrequenti2 = NER(POStag2)
    print()

    print("Segue una lista dei quindici nomi propri di persona più frequenti nel primo testo")
    for i in personePiuFrequenti1:
        print("Il nome", i[0], "con frequenza", i[1])
    print("Segue una lista dei quindici nomi propri di persona più frequenti nel secondo testo")
    for i in personePiuFrequenti2:
        print("Il nome", i[0], "con frequenza", i[1])
    print()

    print("Segue una lista dei quindici nomi propri di luogo più frequenti nel primo testo")
    for i in luoghiPiuFrequenti1:
        print("Il nome", i[0], "con frequenza", i[1])
    print("Segue una lista dei quindici nomi propri di luogo più frequenti nel secondo testo")
    for i in luoghiPiuFrequenti2:
        print("Il nome", i[0], "con frequenza", i[1])

main(sys.argv[1], sys.argv[2])
